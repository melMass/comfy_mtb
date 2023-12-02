## Core
These 3 scripts cannot be used independently and must all be present to work, they are mostly enhancing the frontend of python nodes
- `comfy_shared`: library of methods used in `mtb_widgets` and `debug`

**mtb_widgets** define ui callbacks, and various widgets like the `COLOR` type:  
<img src="https://github.com/melMass/comfy_mtb/assets/7041726/5dbcb714-e1e2-4be7-b0e2-68a6c38c83de" width=400/>

or the `BOOL` type:  
<img src="https://github.com/melMass/comfy_mtb/assets/7041726/7601366d-601c-4f4d-b735-1a4b076770b0" width=400/>

There is also `Debug` which is a node that should be able to display any data input, it handle a few cases and fallback to the string representation of the
data otherwise:
![debug](https://github.com/melMass/comfy_mtb/assets/7041726/1f4393e4-1c3d-4807-9501-fe8888bfae25)


**note +**
A basic HTML note mainly to add better looking notes/instructions for workflow makers:
![image](https://github.com/melMass/comfy_mtb/assets/7041726/2ba1f832-0044-4bad-974c-e6387981af57)


## Standalone
These scripts can be taken and placed independently of `comfy_mtb` or any other files, mimicking what pythongosss did for their 

- **imageFeed**: a fork of @pythongosssss ' s [image feed](https://github.com/pythongosssss/ComfyUI-Custom-Scripts/tree/main/js), it adds support for: a lightbox to see images bigger, a way to load the current session history (in case of a web page reload), and different icons, most of the work come from the original script. 


> **NOTE**
> 
> The original imagefeed got updated since and offer more options, ideally I would clean my lightbox thing and PR it to pythongoss later but in the meantime the script will detect if you already use the original one and not load this fork


- ![imagefeed2-hd](https://github.com/melMass/comfy_mtb/assets/7041726/8539f46f-78e1-459a-a11c-fddd44e63ca9)


- **notify**: a basic toast notification system that I use in some places accross mtb, it can be used by simply calling `window.MTB.notify("Hello world!")`  
![extract](https://github.com/melMass/comfy_mtb/assets/7041726/450c67fc-a7e9-4bea-ae49-b610d693098d)
