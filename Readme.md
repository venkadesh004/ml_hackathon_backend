The Tflearn used here will have a problem in running
Error: AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

Go to tflearn definition, then to data_utils
In PIL.Image at line 577 change to 
  resize_mode=Image.LANCZOS
