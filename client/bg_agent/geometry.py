"""Client-pixel geometry; viewport origin is applied exactly once by planning."""
from dataclasses import dataclass
import math

from .windows import Box


@dataclass(frozen=True)
class CenteredRow:
    center_x: float
    center_y: float
    pitch: float

    def boxes(self,count,width,height):
        if not isinstance(count,int) or not 0<=count<=10:raise ValueError('invalid_row_count')
        if not all(math.isfinite(v) for v in (self.center_x,self.center_y,self.pitch,width,height)) or min(self.pitch,width,height)<=0:
            raise ValueError('invalid_row_geometry')
        return [Box(round(self.center_x+(i-(count-1)/2)*self.pitch-width/2),
                    round(self.center_y-height/2),round(width),round(height)) for i in range(count)]


def client_to_screen(point,client):
    """Client and capture pixels are physical: never multiply by DPI again."""
    x,y=(client['x'],client['y']) if isinstance(client,dict) else (client.x,client.y)
    return [point[0]+x,point[1]+y]
