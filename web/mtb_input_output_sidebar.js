import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

import * as shared from './comfy_shared.js'

if (window?.__COMFYUI_FRONTEND_VERSION__) {
  const version = window?.__COMFYUI_FRONTEND_VERSION__
  console.log(`%c ${version}`, 'background: orange; color: white;')

  app.extensionManager.registerSidebarTab({
    id: 'mtb-inputs-outputs',
    icon: 'pi pi-images',
    title: 'Input & Outputs',
    tooltip: 'Browse inputs and outputs directories.',
    type: 'custom',
    // this is run everytime the tab's diplay is toggled on.
    render: async (el) => {
      const inputs = await api.fetchApi('/mtb/actions', {
        method: 'POST',
        body: JSON.stringify({
          name: 'getInputs',
        }),
      })
      const output = await inputs.json()
      const urls = output?.result
      if (!urls) return

      const cont = document.createElement('div')
      Object.assign(cont, 'style', {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        justifyContent: 'flex-start',
      })

      el.appendChild(cont)

      for (const [key, url] of Object.entries(urls)) {
        const a = document.createElement('img')
        a.src = url
        a.width = 200
        cont.appendChild(a)

        // cont.appendChild(document.createElement('br'))
      }

      // el.innerHTML = inputs.join('\n') // .map((p) => `<div>${p}</div>`).join('\n')
    },
  })
}
