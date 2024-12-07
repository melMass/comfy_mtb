// NOTE: this will be the LT part of mtb API system
// I need to properly publish the source and fix a few things before

// import { app } from '../../scripts/app.js'
// // import { api } from '../../scripts/api.js'
//
// import * as shared from './comfy_shared.js'
// import { createOutliner } from './dist/mtb_inspector.js'
//
// if (window?.__COMFYUI_FRONTEND_VERSION__) {
//   const version = window?.__COMFYUI_FRONTEND_VERSION__
//   console.log(`%c ${version}`, 'background: orange; color: white;')
//
//   const panel = app.extensionManager.registerSidebarTab({
//     id: 'mtb-nodes',
//     icon: 'pi pi-bolt',
//     title: 'MTB',
//     tooltip: 'MTB: API outliner',
//     type: 'custom',
//     // this is run everytime the tab's diplay is toggled on.
//     render: (el) => {
//       const outliner = createOutliner(el)
//       const inputs = shared.getAPIInputs()
//       console.log('INPUTS', inputs)
//       outliner.$$set({ inputs })
//     },
//   })
// }
