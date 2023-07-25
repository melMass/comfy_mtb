## Core
These 3 script should cannot be used independently and must all be present to work
- `comfy_shared`: library of methods used in `mtb_widgets` and `debug`

## Standalone
These scripts can be taken and placed independently of `comfy_mtb` or any other files, mimicking what pythongosss did for their [Custom Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts/tree/main/js)

- **imageFeed**: a fork of pythongosssss's image feed, it adds support for: a lightbox to see images bigger, a way to load the current session history (in case of a web page reload), and different icons, most of the work come from the original script.  
- ![imagefeed](https://github.com/melMass/comfy_mtb/assets/7041726/c3acab4b-d28b-4432-a31b-248391aa2ee8)

- **notify**: a basic toast notification system that I use in some places accross mtb, it can be used by simply calling `window.MTB.notify("Hello world!")`  
![extract](https://github.com/melMass/comfy_mtb/assets/7041726/450c67fc-a7e9-4bea-ae49-b610d693098d)
