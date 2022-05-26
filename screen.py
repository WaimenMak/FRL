from pyglet.gl import *

# import pyglet

# window = pyglet.window.Window()
# from xvfbwrapper import Xvfb
# # from pyvirtualdisplay import Display
# import time
# import os
# os.system("bash xvfb-run.sh -s "-screen 0 1400x900x24"")
# # display = Display(visible=0, size=(900, 800))
# # display.start()

# vdisplay = Xvfb()
# vdisplay.start()

# try:
# 	print (10)
# 	time.sleep(10)
#     # launch stuff inside virtual display here.

# finally:
#     # always either wrap your usage of Xvfb() with try / finally,
#     # or alternatively use Xvfb as a context manager.
#     # If you don't, you'll probably end up with a bunch of junk in /tmp
#     vdisplay.stop()
#     print("yes")
#     # display.stop()