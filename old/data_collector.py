# pip install pyscreenshot
# pip install Pillow

import pyscreenshot as ImageGrab
import time
images_folder="captured_images/0/"
 
for i in range(0,20):
   time.sleep(4)
   im=ImageGrab.grab(bbox=(120,410,800,1000)) #x1,y1,x2,y2
   print("saved......",i)
   im.save(images_folder+str(i)+'.png')
   print("clear screen now and redraw now........")



# or use MS paint and manually create them   
# this is what i did .