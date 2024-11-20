import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

// import * as shared from './comfy_shared.js'

import {
  // defineCSSClass,
  ensureMTBStyles,
  makeElement,
  makeSelect,
  makeSlider,
  renderSidebar,
} from './mtb_ui.js'

let offset = 0
let currentWidth = 200
let currentMode = 'input'
let currentSort = 'None'

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
    if (currentMode === 'input') {
      a.onclick = (_e) => {
        const selected = app.canvas.selected_nodes
        if (selected && Object.keys(selected).length === 0) {
          app.extensionManager.toast.add({
            severity: 'warn',
            summary: 'No LoadImage node selected!',
            detail:
              'For now the only action when clicking images in the sidebar is to set the image on all selected LoadImage nodes.',
            life: 5000,
          })
          return
        }

        for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
          updateImage(node, `${key}.png`)
        }
      }
    } else {
      a.onclick = (_e) =>
        // window.MTB?.notify?.("Output import isn't supported yet...", 5000)
        app.extensionManager.toast.add({
          severity: 'warn',
          summary: 'Outputs not supported',
          detail:
            'For now only inputs can be clicked to load the image on the active LoadImage node.',
          life: 5000,
        })
    }
    imgs.push(a)
  }
  if (target !== undefined) {
    target.append(...imgs)
  }
  return imgs
}

const getUrls = async () => {
  const count = await api.getSetting('mtb.io-sidebar.count')
  console.log('Sidebar count', count)
  const inputs = await api.fetchApi('/mtb/actions', {
    method: 'POST',
    body: JSON.stringify({
      name: 'getUserImages',
      // mode, count, offset
      args: [currentMode, count, offset, currentSort],
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

  app.ui.settings.addSetting({
    id: 'mtb.io-sidebar.count',
    category: ['mtb', 'Input & Output Sidebar', 'count'],

    name: 'Number of images to fetch',
    type: 'number',
    defaultValue: 1000,

    tooltip:
      "This setting affects the input/output sidebar to determine how many images to fetch per pagination (pagination is not yet supported so for now it's the static total)",
    attrs: {
      style: {
        // fontFamily: 'monospace',
      },
    },
  })

  app.ui.settings.addSetting({
    id: 'mtb.io-sidebar.img-size',
    category: ['mtb', 'Input & Output Sidebar', 'img-size'],

    name: 'Resolution of the images',
    type: 'number',
    defaultValue: 512,

    tooltip: "It's recommended to keep it at 512px",
    attrs: {
      style: {
        // fontFamily: 'monospace',
      },
    },
  })
  app.ui.settings.addSetting({
    id: 'mtb.io-sidebar.sort',
    category: ['mtb', 'Input & Output Sidebar', 'sort'],
    name: 'Default sort mode',
    type: 'combo',

    onChange: (v) => {
      // alert(`Sort is now ${v}`)
      currentSort = v
    },

    defaultValue: 'Modified',
    // tooltip: "It's recommended to keep it at 512px",
    options: ['None', 'Modified', 'Modified-Reverse', 'Name', 'Name-Reverse'],
  })

  app.extensionManager.registerSidebarTab({
    id: 'mtb-inputs-outputs',
    icon: 'pi pi-images',
    title: 'Input & Outputs',
    tooltip: 'MTB: Browse inputs and outputs directories.',
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
          const urls = await getUrls()
          if (urls) {
            imgs = getImgsFromUrls(urls, imgGrid)
          }
        }
      })

      const imgTools = makeElement('div.mtb_tools')
      const orderSelect = makeSelect(
        ['None', 'Modified', 'Modified-Reverse', 'Name', 'Name-Reverse'],
        currentSort,
      )

      orderSelect.addEventListener('change', async (e) => {
        const newSort = e.target.value
        const changed = newSort !== currentSort
        currentSort = newSort
        if (changed) {
          imgGrid.innerHTML = ''
          const urls = await getUrls()
          if (urls) {
            imgs = getImgsFromUrls(urls, imgGrid)
          }
        }
      })

      const sizeSlider = makeSlider(64, 1024, currentWidth, 1)
      imgTools.appendChild(orderSelect)

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
