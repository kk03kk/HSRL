"""Windows control surface for the local Battlegrounds runner."""
import ctypes
from ctypes import wintypes
from datetime import datetime
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
from threading import Thread
import time
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsrl_agent.live.contracts import Blocked
from hsrl_agent.live.operator_control import read_state
from hsrl_agent.live.runtime import exclusive, load, save
from hsrl_agent.live.selection import selected_checkpoint
from hsbg.runtime.paths import workspace_asset


GAME = Path(os.environ.get('HSAI_GAME_DIR', r'D:\Hearthstone'))
PYTHON = Path(sys.executable).with_name('python.exe')
BG, PANEL, BORDER = '#101722', '#1a2534', '#29394b'
TEXT, MUTED, GOLD = '#f3f5f7', '#a9bacd', '#f5c773'
GREEN, BLUE, RED = '#71d7b6', '#91befb', '#ff9f9f'
FONT = 'Microsoft YaHei UI'


def desktop_checkpoints(root, config):
    pin = load(root / 'config/live/model.json')
    if pin.get('action_profile') != config['action_profile']:
        raise ValueError('控制台与模型的动作规则不一致')
    return {
        config['default_label']: selected_checkpoint(root),
        config['fallback_label']: workspace_asset(root, pin['fallback_checkpoint']),
    }


STATES = {
    'idle': ('待启动', MUTED), 'starting': ('正在启动', BLUE),
    'running': ('游戏中', GREEN), 'waiting': ('等待中', BLUE),
    'pausing': ('正在收尾', GOLD), 'paused': ('已暂停', GOLD),
    'stopping': ('正在安全停止', GOLD), 'stopped': ('已停止', MUTED),
    'reconnecting': ('正在重连', GOLD), 'finished': ('已完成', GREEN),
    'error': ('需要检查', RED),
}
DETAILS = {
    'preparing': '正在验证炉石窗口并加载模型',
    'lobby': '正在确认单人酒馆战棋大厅',
    'matchmaking': '炉石正在寻找对手',
    'match': '已连接真实对局',
    'next_turn': '等待客户端进入下一回合',
    'player150_budget': '已用完本回合的 150 步预算，等待炉石结算',
    'foreground': '请将炉石切回前台',
    'between_matches': '已暂停，恢复后才继续匹配',
    'match_finished': '已记录对局结果，准备返回大厅',
    'return_lobby': '目标局数已完成，正在返回大厅',
    'receipt_or_continuation': '等待当前操作的目标、确认和回执',
    'log_rebuild': '正在读取炉石日志',
    'finalizing': '正在保存运行证据',
    'game_server': '等待炉石客户端的重连和状态确认',
    'historical_receipt': '上一次运行仍有未确认操作，等待日志核对',
    'action point occluded': '点击位置被遮挡，请关闭挡住炉石的浮层',
    'client connection is not confirmed connected': '正在确认游戏服务器连接',
    'reconnect could not be verified; input stopped and ledger retained': '重连未获确认，已停止输入，请检查炉石',
    'waiting for visible hero row; no hero input sent': '等待英雄选择画面稳定',
    'not a decision phase': '等待下一个可操作阶段',
    'server snapshot or visual playback is pending': '等待游戏状态与画面同步',
    'client combat playback has not reached server recruit; input paused': '等待战斗播放结束',
}


def listen_hotkey(events):
    """Own the global chord on a dedicated Win32 message queue."""
    user = ctypes.windll.user32
    hotkey_id = 1
    if not user.RegisterHotKey(None, hotkey_id, 0x4000 | 0x0002 | 0x0001, ord('P')):
        events.put(('hotkey_status', False))
        return
    events.put(('hotkey_status', True))
    message = wintypes.MSG()
    try:
        while user.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == 0x0312 and message.wParam == hotkey_id:
                events.put(('hotkey', None))
    finally:
        user.UnregisterHotKey(None, hotkey_id)


class Console:
    def __init__(self, window):
        self.window = window
        self.events = SimpleQueue()
        self.process = None
        self.folder = None
        self.sequence = 0
        self.command = 'run'
        self.next_heartbeat = 0.
        self.exiting = False
        self.snapshot = {}
        self.completed = 0
        self.target = 1
        self.last_error = ''
        self.last_log = ''
        self.mode = tk.StringVar(value='single')
        self.config = load(ROOT / 'config/live/desktop-models.json')
        if self.config.get('schema') != 'hsai.desktop-models/v1':
            raise ValueError('控制台模型配置无效')
        self.models = desktop_checkpoints(ROOT, self.config)
        self.model_name = tk.StringVar(value=next(iter(self.models)))
        self.count = tk.StringVar(value='2')
        self.shortcut = tk.StringVar(value='正在注册全局快捷键')
        self.create_widgets()
        Thread(target=listen_hotkey, args=(self.events,), daemon=True).start()
        window.after(100, self.refresh)
        window.protocol('WM_DELETE_WINDOW', self.close)

    def label(self, parent, text, *, color=TEXT, size=10, bold=False, bg=PANEL, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=(FONT, size, 'bold' if bold else 'normal'), **kwargs)

    def create_widgets(self):
        w = self.window
        w.title('HSAI · 酒馆战棋控制台')
        w.configure(bg=BG)
        w.resizable(False, False)
        style = ttk.Style(w)
        style.theme_use('clam')
        style.configure('Control.TButton', background=BORDER, foreground=TEXT,
                        borderwidth=0, padding=(12, 8), font=(FONT, 10, 'bold'))
        style.map('Control.TButton', background=[('active', '#35495e')],
                  foreground=[('disabled', '#8091a4')])
        style.configure('Start.TButton', background=GOLD, foreground=BG,
                        borderwidth=0, padding=(17, 8), font=(FONT, 10, 'bold'))
        style.map('Start.TButton', background=[('disabled', BORDER), ('active', '#ffdc9b')],
                  foreground=[('disabled', '#8091a4')])
        style.configure('Mode.TRadiobutton', background=PANEL, foreground=TEXT, font=(FONT, 10),
                        focuscolor=PANEL, indicatorcolor=BG, indicatormargin=5)
        style.map('Mode.TRadiobutton', background=[('active', PANEL)],
                  indicatorcolor=[('selected', GOLD)], foreground=[('disabled', MUTED)])
        style.configure('Count.TSpinbox', fieldbackground=BG, foreground=TEXT, arrowsize=12,
                        background=BORDER, bordercolor=BORDER, insertcolor=TEXT)
        style.configure('Games.Horizontal.TProgressbar', troughcolor=BG, background=GOLD,
                        bordercolor=BG, lightcolor=GOLD, darkcolor=GOLD, thickness=5)

        body = tk.Frame(w, bg=BG, padx=24, pady=19)
        body.pack(fill='both', expand=True)
        header = tk.Frame(body, bg=BG)
        header.pack(fill='x', pady=(0, 16))
        self.label(header, '酒馆战棋控制台', bg=BG, size=19, bold=True).pack(side='left')
        self.label(header, 'PLAYER150  /  单人', bg=BG, color=GOLD, size=10, bold=True).pack(side='right')

        status = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=13)
        status.pack(fill='x')
        status_row = tk.Frame(status, bg=PANEL)
        status_row.pack(fill='x')
        self.status = self.label(status_row, '', size=15, bold=True)
        self.status.pack(side='left')
        self.score = self.label(status_row, '', color=MUTED, size=11)
        self.score.pack(side='right')
        self.detail = self.label(status, '', color=MUTED, anchor='w', justify='left')
        self.detail.pack(fill='x', pady=(3, 9))
        self.bar = ttk.Progressbar(status, style='Games.Horizontal.TProgressbar')
        self.bar.pack(fill='x')

        operations = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=12)
        operations.pack(fill='x', pady=11)
        model_row = tk.Frame(operations, bg=PANEL)
        model_row.pack(fill='x', pady=(0, 9))
        self.label(model_row, '模型', color=MUTED).pack(side='left', padx=(0, 10))
        self.model_select = ttk.Combobox(model_row, textvariable=self.model_name, values=list(self.models),
                                         state='readonly', font=(FONT,10))
        self.model_select.pack(side='left', fill='x', expand=True)
        modes = tk.Frame(operations, bg=PANEL)
        modes.pack(fill='x', pady=(0, 10))
        self.modes = []
        for text, value in (('单局', 'single'), ('指定局数', 'count'), ('连续运行', 'infinite')):
            option = ttk.Radiobutton(modes, text=text, value=value, variable=self.mode,
                                     style='Mode.TRadiobutton', command=self.update_buttons)
            option.pack(side='left', padx=(0, 11))
            self.modes.append(option)
            if value == 'count':
                self.spinner = ttk.Spinbox(modes, textvariable=self.count, width=4, from_=2, to=999,
                                           style='Count.TSpinbox', font=(FONT, 10))
                self.spinner.pack(side='left', padx=(0, 12))
        controls = tk.Frame(operations, bg=PANEL)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='开始匹配', command=self.start, style='Start.TButton')
        self.pause_button = ttk.Button(controls, text='暂停', command=self.toggle_pause, style='Control.TButton')
        self.stop_button = ttk.Button(controls, text='安全停止', command=self.stop, style='Control.TButton')
        for button in (self.start_button, self.pause_button, self.stop_button):
            button.pack(side='left', padx=(0, 8))
        self.label(controls, '', color=MUTED, textvariable=self.shortcut, size=9).pack(side='right')

        logs = tk.Frame(body, bg=BG)
        logs.pack(fill='x')
        self.label(logs, '运行日志', bg=BG, bold=True).pack(side='left')
        self.log_button = ttk.Button(logs, text='打开日志', command=self.open_logs, style='Control.TButton')
        self.log_button.pack(side='right')
        log_panel = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        log_panel.pack(fill='x', pady=(5, 0))
        self.log = tk.Text(log_panel, width=67, height=7, state='disabled', bg=PANEL, fg=MUTED,
                           relief='flat', padx=9, pady=7, wrap='word', font=(FONT, 9),
                           selectbackground=BORDER, selectforeground=TEXT, highlightthickness=0)
        self.log.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(log_panel, command=self.log.yview)
        scroll.pack(side='right', fill='y')
        self.log.configure(yscrollcommand=scroll.set)
        self.show('idle', '先在炉石中打开单人酒馆战棋匹配页')
        self.update_buttons()

    def show(self, state, detail='', turn=None):
        name, color = STATES.get(state, STATES['waiting'])
        self.status.configure(text='●  ' + name, fg=color)
        translated = DETAILS.get(detail, detail)
        if detail.startswith('action-dependent card layout is still moving'):
            translated = '等待卡牌动画稳定'
        if state == 'paused' and detail in ('match', 'lobby', 'matchmaking', 'log_rebuild'):
            translated = 'AI 已停止输入，炉石仍会继续计时'
        if turn:
            translated = f'回合 {turn}  ·  {translated}'
        self.detail.configure(text=translated[:160], wraplength=self.window.winfo_fpixels('5.5i'))
        self.score.configure(text=f'已完成  {self.completed} / {self.target or "∞"}')
        self.bar.configure(maximum=self.target or 1, value=self.completed if self.target else 0)

    def update_buttons(self):
        live = self.process is not None
        self.start_button.configure(state='disabled' if live else 'normal')
        self.model_select.configure(state='disabled' if live else 'readonly')
        for option in self.modes:
            option.configure(state='disabled' if live else 'normal')
        self.spinner.configure(state='normal' if not live and self.mode.get() == 'count' else 'disabled')
        paused = self.snapshot.get('state') == 'paused' and self.snapshot.get('sequence') == self.sequence
        can_toggle = live and (self.command == 'run' or (self.command == 'pause' and paused))
        self.pause_button.configure(text='恢复' if self.command == 'pause' else '暂停',
                                    state='normal' if can_toggle else 'disabled')
        self.stop_button.configure(state='normal' if live and self.command != 'stop' else 'disabled')
        self.log_button.configure(state='normal' if self.folder else 'disabled')

    def note(self, line):
        if not line or line == self.last_log:
            return
        self.last_log = line
        self.log.configure(state='normal')
        self.log.insert('end', datetime.now().strftime('%H:%M:%S  ') + line.rstrip()[:700] + '\n')
        if int(self.log.index('end-1c').split('.')[0]) > 220:
            self.log.delete('1.0', '41.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def send(self, command=None):
        if command is not None:
            self.command = command
            self.sequence += 1
        save(self.folder / 'command.json', dict(command=self.command, sequence=self.sequence, at=time.time()))
        self.next_heartbeat = time.monotonic() + 2.

    def start(self):
        if self.process is not None:
            return
        try:
            target = {'single': 1, 'infinite': 0}.get(self.mode.get())
            if target is None:
                target = int(self.count.get())
                if not 2 <= target <= 999:
                    raise ValueError('指定局数请输入 2–999')
            checkpoint = self.models[self.model_name.get()]
            engine = ROOT / self.config['engine']
            for path in (engine, checkpoint, GAME / 'Hearthstone.exe', PYTHON):
                if not path.exists():
                    raise FileNotFoundError(path)
            run_id = f'live-control-{datetime.now():%Y%m%d}-{datetime.now():%H%M%S}-{uuid.uuid4().hex[:5]}'
            self.folder = ROOT / 'artifacts/runs' / run_id
            self.folder.mkdir(parents=True, exist_ok=False)
            self.sequence = 0
            self.send('run')
            self.completed, self.target = 0, target
            self.snapshot = {}
            self.last_error = ''
            command = [str(PYTHON), '-X', 'utf8', '-u', str(ROOT / 'client/live_245.py'), 'autoplay',
                       '--desktop-control', '--enable-input', '--timing', 'fast',
                       '--checkpoint', str(checkpoint), '--action-profile', self.config['action_profile'],
                       '--game-dir', str(GAME), '--out', str(self.folder),
                       '--games', str(target), '--inference-threads', '8', '--seconds', '0']
            env = dict(os.environ, PYTHONPATH=str(engine))
            self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NO_WINDOW)
            Thread(target=self.collect, args=(self.process, self.folder), daemon=True).start()
            self.note('正在加载 ' + self.model_name.get())
            self.show('starting', 'preparing')
        except (OSError, ValueError) as exc:
            self.show('error', str(exc))
            self.note(str(exc))
        self.update_buttons()

    def collect(self, process, folder):
        log = None
        try:
            log = (folder / 'process.log').open('a', encoding='utf-8', buffering=1)
        except OSError as exc:
            self.events.put(('log', str(exc)))
        with process.stdout as stream:
            for line in stream:
                if log:
                    try:
                        log.write(line)
                    except OSError:
                        log.close()
                        log = None
                self.events.put(('log', line.rstrip()))
        if log:
            log.close()
        self.events.put(('exit', process.wait()))

    def request(self, command):
        try:
            self.send(command)
        except OSError as exc:
            self.next_heartbeat = 0.
            self.note(f'控制请求暂未送达，正在重试  {exc}')
        self.update_buttons()

    def toggle_pause(self):
        if not self.process or self.command == 'stop':
            return
        if self.command == 'run':
            self.request('pause')
            self.show('pausing', 'receipt_or_continuation')
        elif self.snapshot.get('state') == 'paused' and self.snapshot.get('sequence') == self.sequence:
            self.request('run')
            self.show('waiting', '正在重新确认炉石状态')

    def stop(self):
        if self.process and self.command != 'stop':
            self.request('stop')
            self.show('stopping', 'receipt_or_continuation')

    def refresh(self):
        if self.process:
            if time.monotonic() >= self.next_heartbeat:
                try:
                    self.send()
                except OSError as exc:
                    self.next_heartbeat = time.monotonic() + .5
                    self.note(f'控制消息写入失败  {exc}')
            state = read_state(self.folder / 'control-status.json')
            if isinstance(state, dict) and state != self.snapshot:
                self.snapshot = state
                self.completed = state['completed']
                if state['sequence'] == self.sequence or state['state'] in ('error', 'stopped', 'finished'):
                    self.show(state['state'], state['detail'], state['turn'])
                self.update_buttons()
        for _ in range(80):
            try:
                event, value = self.events.get_nowait()
            except Empty:
                break
            if event == 'hotkey_status':
                self.shortcut.set('Ctrl + Alt + P' if value else '快捷键被占用，请使用按钮')
            elif event == 'hotkey':
                self.toggle_pause()
            elif event == 'log':
                if value.startswith(('BLOCKED:', 'Traceback', 'Run finalization failed')):
                    self.last_error = value
                routine = value in ('waiting_receipt', 'sent', 'applied', 'waiting_next_turn')
                if not routine and '[RapidOCR]' not in value and not value.startswith('action-dependent card layout'):
                    self.note(DETAILS.get(value, value))
            elif event == 'exit':
                self.process = None
                end = read_state(self.folder / 'control-status.json') or {}
                self.completed = end.get('completed', self.completed)
                if value:
                    self.show('error', end.get('detail') or self.last_error or f'执行器退出码 {value}')
                else:
                    self.show(end.get('state', 'stopped'), '本次运行记录已保存')
                self.note(f'执行器已退出（{value}）')
                self.update_buttons()
                if self.exiting:
                    self.window.destroy()
                    return
        self.window.after(100, self.refresh)

    def open_logs(self):
        if self.folder:
            try:
                os.startfile(self.folder)
            except OSError as exc:
                self.note(str(exc))

    def close(self):
        if self.process:
            self.exiting = True
            self.stop()
        else:
            self.window.destroy()


def main():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        pass
    try:
        with exclusive(ROOT / 'artifacts/runs/live-console.lock'):
            window = tk.Tk()
            try:
                Console(window)
            except Exception as error:
                messagebox.showerror('HSAI 控制台启动失败', str(error), parent=window)
                return 1
            window.mainloop()
    except Blocked:
        ctypes.windll.user32.MessageBoxW(None, '控制台已打开，请从任务栏切回已有窗口。', 'HSAI', 0x40)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
