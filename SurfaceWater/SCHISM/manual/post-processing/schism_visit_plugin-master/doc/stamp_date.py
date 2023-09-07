import Image, ImageDraw, ImageFont
import datetime
from dateutil.parser import parse
import os

class DynamicDateTime(object):
    
    def __init__(self, base_datetime,frame_step,loc_rect=(0.4,0.4),
                 font=ImageFont.truetype("arial.ttf", 15),color="black"):
        
        if type(base_datetime) == type(datetime.datetime(2012,1,1)):
            self._start_time = base_datetime
        else:
            self._start_time = parse(base_datetime)
            
        self._step = frame_step
        
        self._font = font
        self._color = color
        self._loc  = loc_rect
                
        
    def color(self,frame):
        return self._color
        
    def font(self,frame):
        return self._font
    
    def location(self,frame):
        return self._loc    
    
    def text(self,frame):
        frame_time = self._start_time+datetime.timedelta(seconds=self._step*frame)
        return frame_time.ctime()
        
class ImageSquence(object):
    
    def __init__(self,image_files_pattern,first_frame,last_frame,loc):
        self._loc = loc
        self._first_frame = first_frame
        self._last_frame  = last_frame
        self._image_pattern =image_files_pattern
        self._first_frame=first_frame
        self._last_frame=last_frame
       
    def location(self,frame):
        return self._loc
        
        
    def image(self,frame):
        if (frame>self._last_frame) or (frame<self._first_frame):
            print "input frame is not available in image sequence"      
        image_file = self._image_pattern%(frame)
        if not(os.path.isfile(image_file)):
                print "warning:"+image_file+" is not valid"
        return Image.open(image_file)


def decorate_image(image_pattern,first_frame,last_frame,
                   dynamic_text=None,dynamic_image=None):
    for index in range(first_frame,last_frame+1):
        image_file = image_pattern%(index)
        if not(os.path.isfile(image_file)):
            print "warning:"+image_file+" is not valid"
            break
        a_image = Image.open(image_file)
        draw  = ImageDraw.Draw(a_image)
     
        if dynamic_text:    
            loc   = dynamic_text.location(index)
            loc_x0 = int(a_image.size[0]*loc[0])
            loc_y0 = int(a_image.size[1]*loc[1])
            font =dynamic_text.font(index)
            color =dynamic_text.color(index)
            text  =dynamic_text.text(index)
            draw.text((loc_x0,loc_y0),text,fill=color,font=font)
            
        if dynamic_image:    
            loc   = dynamic_image.location(index)
            paste_image =dynamic_image.image(index)
            loc_x0 = int(a_image.size[0]*loc[0])
            loc_y0 = int(a_image.size[1]*loc[1])
            loc_x1 = loc_x0+paste_image.size[0]
            loc_y1 = loc_y0+paste_image.size[1]
            paste_loc=(loc_x0,loc_y0,loc_x1,loc_y1)
            a_image.paste(paste_image,paste_loc)
            a_image.save(image_file )
            
            
if __name__=="__main__":
    origin_image = "./base_vel/fract_jun_no_barrier_hvel%04d.png"
    last_frame   = 454
    first_frame  = 0
    base_datetime = parse("06/01/2014")
    frame_step =3600
    paste_image="./tide_plots_jun/tide_%04d.jpeg"
    image_loc = [0.0,0.04,0.1,0.14]
    
    text_font=ImageFont.truetype("arial.ttf", 32)
    text_color="black"
    text_loc  =[0.45,0.1]
    
    datetime_paste = DynamicDateTime(base_datetime,frame_step,
                                      text_loc,text_font,text_color)
                                      
    image_paste= ImageSquence(paste_image,first_frame,last_frame,image_loc)
    
    decorate_image(origin_image,first_frame,last_frame,
                   datetime_paste,image_paste)