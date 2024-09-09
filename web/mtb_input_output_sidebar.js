import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

import * as shared from './comfy_shared.js'

import {
  defineCSSClass,
  ensureMTBStyles,
  makeElement,
  makeSelect,
  makeSlider,
  renderSidebar,
} from './mtb_ui.js'

let currentWidth = 200
let currentMode = 'input'

const IMAGE_NODES = ['LoadImage']

const updateImage = (node, image) => {
  if (IMAGE_NODES.includes(node.type)) {
    const w = node.widgets?.find((w) => w.name === 'image')
    if (w) {
      w.value = image
      w.callback()
    }
  }
}

const getImgsFromUrls = (urls, target) => {
  const imgs = []
  if (urls === undefined) {
    return imgs
  }

  for (const [key, url] of Object.entries(urls)) {
    const a = makeElement('img')
    a.src = url
    a.width = currentWidth
    a.onclick = (e) => {

      for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
        updateImage(node, `${key}.png`)
      }
    }
    imgs.push(a)
  }
  if (target !== undefined) {
    target.append(...imgs)
  }
  return imgs
}

const getUrls = async (mode) => {
  const inputs = await api.fetchApi('/mtb/actions', {
    method: 'POST',
    body: JSON.stringify({
      name: 'getUserImages',
      args: [mode],
    }),
  })
  const output = await inputs.json()
  return output?.result || {}
}

if (window?.__COMFYUI_FRONTEND_VERSION__) {
  let handle
  const version = window?.__COMFYUI_FRONTEND_VERSION__
  console.log(`%c ${version}`, 'background: orange; color: white;')

  ensureMTBStyles()

  app.extensionManager.registerSidebarTab({
    id: 'mtb-inputs-outputs',
    icon: 'pi pi-images',
    title: 'Input & Outputs',
    tooltip: 'Browse inputs and outputs directories.',
    type: 'custom',

    // this is run everytime the tab's diplay is toggled on.
    render: async (el) => {
      if (handle) {
        handle.unregister()
        handle = undefined
      }

      if (el.parentNode) {
        el.parentNode.style.overflowY = 'clip'
      }

      const urls = await getUrls(currentMode)
      let imgs = {}

      const cont = makeElement('div.mtb_sidebar')

      const imgGrid = makeElement('div.mtb_img_grid')
      const selector = makeSelect(['input', 'output'], currentMode)

      selector.addEventListener('change', async (e) => {
        const newMode = e.target.value
        const changed = newMode !== currentMode
        currentMode = newMode
        if (changed) {
          imgGrid.innerHTML = ''
          const urls = await getUrls(currentMode)
          if (urls) {
            imgs = getImgsFromUrls(urls, imgGrid)
          }
        }
      })

      const imgTools = makeElement('div.mtb_tools')
      const sizeSlider = makeSlider(64, 1024, currentWidth, 1)
      imgTools.appendChild(sizeSlider)

      imgs = getImgsFromUrls(urls, imgGrid)

      sizeSlider.addEventListener('input', (e) => {
        currentWidth = e.target.value
        for (const img of imgs) {
          img.style.width = `${e.target.value}px`
        }
      })
      handle = renderSidebar(el, cont, [selector, imgGrid, imgTools])
    },
    destroy: () => {
      if (handle) {
        handle.unregister()
        handle = undefined
      }
    },
  })
}
