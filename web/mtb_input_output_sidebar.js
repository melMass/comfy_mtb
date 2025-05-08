/// <reference path="../types/typedefs.js" />

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'
import { infoLogger, successLogger, errorLogger } from './comfy_shared.js'

import * as shared from './comfy_shared.js'

import {
  // defineCSSClass,
  ensureMTBStyles,
  makeElement,
  makeSelect,
  makeSlider,
  renderSidebar,
  ContextMenu,
} from './mtb_ui.js'

let currentAbortController = null

/** cursor/offset of where we are at */
const offset = 0

/** width of the images in the grid */
let currentWidth = 200

let currentMode = 'input'
let subfolder = ''
let currentSort = 'None'

let clientOnce = false

/** reference to the dom element receiving the images */
let imgGrid = undefined

/** currently loaded image (as object urls) */
let loaded_images = undefined

/**
 * stores the user's full local path to input/output directory
 * This is then used to feed VHS Load Image (from path)
 */
let userDirectories = undefined

// const IMAGE_NODES = ['LoadImage', 'VHS_LoadImagePath']
const VIDEO_NODES = ['VHS_LoadVideo']
const PROCESSED_PROMPT_IDS = new Set()


let contextMenu = undefined

function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      infoLogger('Debouncing method')
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

const debouncedGetUrls = async (ms = 250) => {
  if (loaded_images === undefined) {
    return await getUrls(subfolder)
  }
  debounce(async (subfolder) => {
    const urls = await getUrls(subfolder)
    infoLogger('Loaded URLs (debounced): ', urls)
    if (urls) {
      loaded_images = await getImgsFromUrls(urls, imgGrid)
      infoLogger('Loaded Images (debounced): ', loaded_images)
    }
  }, ms)

  return loaded_images
}

/** Callback on clicking an image in the grid */
const updateImage = (node, image) => {
  switch (node.type) {
    case 'LoadImage': {
      if (subfolder && subfolder !== '') {
        app.extensionManager.toast.add({
          severity: 'warn',
          summary: 'Subfolder not supported',
          detail: "The LoadImage node doesn't support subfolders",
          life: 5000,
        })
        return
      }
      if (currentMode === 'output') {
        app.extensionManager.toast.add({
          severity: 'warn',
          summary: 'Outputs not supported',
          detail:
            "The LoadImage node doesn't support loading outputs, use VHS Load Image Path and I'll resolve the full path.",
          life: 5000,
        })
        return
      }
      // if (IMAGE_NODES.includes(node.type)) {
      const w = node.widgets?.find((w) => w.name === 'image')
      if (w) {
        w.value = image
        w.callback()
      }
      //}
      break
    }
    case 'VHS_LoadImagePath': {
      let value = image

      if (!userDirectories?.output) {
        app.extensionManager.toast.add({
          severity: 'warn',
          summary: 'User output directory not resolved',
          detail: "We couldn't resolve the image full path.",
          life: 5000,
        })
        return
      }

      if (subfolder && subfolder !== '') {
        value = `${subfolder}/${image}`
      }
      value = `${userDirectories.output}/${value}`

      const w = node.widgets?.find((w) => w.name === 'image')
      if (w) {
        console.log(w)
        w.value = value
        // TODO: VHS needs explicity value passsed here
        w.callback(value)
      }
      break
    }
    case VIDEO_NODES.includes(node.type): {
      const w = node.widgets?.find((w) => w.name === 'video')
      if (w) {
        node.updateParameters({ filename: image }, true)
      }
      break
    }
    default: {
      console.warn('No method to update', node.type)
    }
  }
}

/**
 * Converts a result item to a request url.
 * @param {ResultItem} resultItem
 * @returns {string} - The request URL.
 */
const resultItemToQuery = (resultItem) =>
  [
    `/mtb/view?filename=${resultItem.filename}`,
    `width=512`,
    `type=${resultItem.type}`,
    `subfolder=${resultItem.subfolder}`,
    `preview=`,
  ].join('&')

/**
 * Retrieves the unique prompt ID from a history task item.
 * @param {HistoryTaskItem} historyTaskItem
 * @returns {string} - The prompt ID.
 */
const getPromptId = (historyTaskItem) => `${historyTaskItem.prompt[1]}`

/**
 * Process and return any new/unseen outputs from the most recent history item.
 * @param {HistoryTaskItem} mostRecentTask - The most recent history task item.
 * @returns {Object<string, string>} - A map of task outputs URLs.
 */
const getNewOutputUrls = (mostRecentTask) => {
  if (!mostRecentTask) return

  const promptId = getPromptId(mostRecentTask)
  if (PROCESSED_PROMPT_IDS.has(promptId)) return

  const urls = {}
  for (const nodeOutputs of Object.values(mostRecentTask.outputs)) {
    const { images, audio, animated } = nodeOutputs
    if (images) {
      const imageOutputs = Object.values(nodeOutputs.images)
      imageOutputs.forEach(
        (resultItem) =>
          (urls[resultItem.filename] = resultItemToQuery(resultItem))
      )
    }
    // Can process `animated` and `audio` outputs here.
  }

  const foundNewOutputs = Object.keys(urls).length > 0
  if (!foundNewOutputs) return null

  PROCESSED_PROMPT_IDS.add(promptId)
  return urls
}

/** Fetch history and update the grid with any new ouput images. */
const updateOutputsGrid = async () => {
  try {
    const history = await api.getHistory(/** maxSize: */ 1)
    const mostRcentTask = history.History[0]
    const newUrls = getNewOutputUrls(mostRcentTask)
    if (newUrls) {
      const imgGrid = document.querySelector('.mtb_img_grid')
      getImgsFromUrls(newUrls, imgGrid, { prepend: true })
    }
  } catch (error) {
    console.error('Error fetching history:', error)
  }
}

const getImgsFromUrls = (urls, target, options = { prepend: false }) => {
  if (currentAbortController) {
    currentAbortController.abort()
  }
  infoLogger('getting images from urls', urls)

  currentAbortController = new AbortController()
  const { signal } = currentAbortController
  const imgs = []
  if (!urls) return imgs

  const loadingIndicator = document.createElement('div')
  loadingIndicator.className = 'mtb-loading-indicator'
  if (target) target.appendChild(loadingIndicator)

  const totalImages = Object.keys(urls).length
  let loadedCount = 0
  const updateLoadingStatus = () => {
    loadingIndicator.textContent = `Loaded ${loadedCount} of ${totalImages} images`
  }
  updateLoadingStatus()

  try {
    const loadImage = async (key, url) => {
      try {
        const response = await fetch(url, { signal })
        if (!response.ok) {
          console.warn(`Failed to fetch ${key}: ${response.status}`)

          return null
        }
        // throw new Error(`HTTP error! status: ${response.status}`)
        const blob = await response.blob()
        const imgUrl = URL.createObjectURL(blob)

        const elem = makeElement(currentMode === 'video' ? 'video' : 'img')
        elem.src = imgUrl
        elem.width = currentWidth

        // cleanup
        elem.onload = () => URL.revokeObjectURL(imgUrl)
        elem.onerror = () => URL.revokeObjectURL(imgUrl)

        // Add click handler for input mode
        // if (currentMode === 'input') {
        // elem.onclick = (_e) => {
        // Your existing click handler code
        // }
        // }

        // Add context menu
        elem.addEventListener('contextmenu', (e) => {
          e.preventDefault()
          const contextMenuItems = [
            {
              label: 'Add Node with Image',
              icon: '🖼',
              action: () => {
                const node = app.graph.createNode('LoadImage')
                updateImage(node, key)
              },
            },
            {
              label: 'Load Workflow from Image',
              icon: '📋',
              action: async () => {
                try {
                  const response = await fetch(url)
                  const data = await response.blob()
                  // Assuming you have a function to extract workflow from image metadata
                  const workflow = await extractWorkflowFromImage(data)
                  if (workflow) {
                    app.loadGraphData(workflow)
                  }
                } catch (error) {
                  app.extensionManager.toast.add({
                    severity: 'error',
                    summary: 'Error',
                    detail: 'Failed to load workflow from image',
                    life: 3000,
                  })
                }
              },
            },
            {
              label: 'View Full Image',
              icon: '🔍',
              action: () => {
                window.open(url, '_blank')
              },
            },
          ]
          contextMenu.show(e.pageX, e.pageY, contextMenuItems, {
            elem,
            key,
            url,
          })
        })

        elem.onclick = (_e) => {
          const selected = app.canvas.selected_nodes
          if (!selected || Object.keys(selected).length === 0) {
            app.extensionManager.toast.add({
              severity: 'warn',
              summary: 'No node selected!',
              detail: 'Please select a node first.',
              life: 5000,
            })
            return
          }

          for (const [_id, node] of Object.entries(selected)) {
            updateImage(node, key)
          }
        }

        loadedCount++
        updateLoadingStatus()

        return elem
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log('Fetch aborted')
          return null
        }
        console.error('Error loading image:', error)
        return null
      }
    }
    const BATCH_SIZE = 20
    for (let i = 0; i < Object.entries(urls).length; i += BATCH_SIZE) {
      const batch = Object.entries(urls).slice(i, i + BATCH_SIZE)
      const loadedImages = await Promise.all(
        batch.map(([key, url]) => loadImage(key, url)),
      )

      const validImages = loadedImages.filter((img) => img !== null)
      imgs.push(...validImages)

      if (target) {
        target.append(...validImages)
      }
    }

    return imgs

    // return
    // const elem = currentMode === 'video' ? 'video' : 'img'

    for (const [key, url] of Object.entries(urls)) {
      const a = makeElement(elem)
      a.src = url
      a.width = currentWidth

      const selected = app.canvas.selected_nodes

      if (currentMode === 'input') {
        a.onclick = (_e) => {
          // if (subfolder !== '') {
          //   app.extensionManager.toast.add({
          //     severity: 'warn',
          //     summary: 'Subfolder not supported',
          //     detail: "The LoadImage node doesn't support subfolders",
          //     life: 5000,
          //   })
          //   return
          // }
          if (selected && Object.keys(selected).length === 0) {
            app.extensionManager.toast.add({
              severity: 'warn',
              summary: 'No node selected!',
              detail:
                'For now the only action when clicking images in the sidebar is to set the image on all selected LoadImage nodes.',
              life: 5000,
            })
            return
          }

          for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
            updateImage(node, key)
          }
        }
      } else if (currentMode === 'output') {
        a.onclick = (_e) => {
          if (selected && Object.keys(selected).length === 0) {
            return
          }
          for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
            updateImage(node, key)
          }

          // window.MTB?.notify?.("Output import isn't supported yet...", 5000)
          // if (subfolder !== '') {
          //   app.extensionManager.toast.add({
          //     severity: 'warn',
          //     summary: 'Subfolder not supported',
          //     detail: "The LoadImage node doesn't support subfolders",
          //     life: 5000,
          //   })
          //   return
          // }
          //
          // app.extensionManager.toast.add({
          //   severity: 'warn',
          //   summary: 'Outputs not supported',
          //   detail:
          //     'For now only inputs can be clicked to load the image on the active LoadImage node.',
          //   life: 5000,
          // })
        }
      } else {
        a.autoplay = true

        a.muted = true
        a.loop = true
        a.onclick = (_e) => {
          const selected = app.canvas.selected_nodes
          if (selected && Object.keys(selected).length === 0) {
            app.extensionManager.toast.add({
              severity: 'warn',
              summary: 'No node selected!',
              detail:
                "For now the only action when clicking videos in the sidebar is to set the video on all selected 'Load Video (Upload)' nodes.",
              life: 5000,
            })
            return
          }

          for (const [_id, node] of Object.entries(app.canvas.selected_nodes)) {
            updateImage(node, key)
          }
        }
      }
      imgs.push(a)
    }
    if (target !== undefined) {
      if (options.prepend) target.prepend(...imgs)
    else target.append(...imgs)
    }
    return imgs
  } finally {
    // Keep loading indicator visible for a moment after completion
    setTimeout(() => {
      if (target && loadingIndicator.parentNode === target) {
        loadingIndicator.remove()
      }
    }, 2000)
  }
}
// Helper function to extract workflow from image metadata
async function extractWorkflowFromImage(blob) {
  // Implementation depends on how the workflow data is stored in the image
  // This is just a placeholder
  try {
    // You might need to use ExifReader or similar library to extract metadata
    return null
  } catch (error) {
    console.error('Failed to extract workflow:', error)
    return null
  }
}

const getModes = async () => {
  const inputs = await shared.runAction('getUserImageFolders')
  return inputs
}
const getUrls = async (subfolder) => {
  const count = (await api.getSetting('mtb.io-sidebar.count')) || 1000
  console.debug('Sidebar count', count)
  if (currentMode === 'video') {
    const output = await shared.runAction(
      'getUserVideos',
      256,
      count,
      offset,
      currentSort,
    )
    return output || {}
  }
  const output = await shared.runAction(
    'getUserImages',
    currentMode,
    count,
    offset,
    currentSort,
    false,
    subfolder,
  )
  return output || {}
}

const build_ui = async (el) => {
  if (el.parentNode) {
    el.parentNode.style.overflowY = 'clip'
  }

  const allModes = await getModes()

  const input_modes = allModes.input.map((m) => `input - ${m}`)
  const output_modes = allModes.output.map((m) => `output - ${m}`)

  if (!userDirectories) {
    userDirectories = {
      input: allModes.input_root,
      output: allModes.output_root,
    }
    infoLogger('User directories', userDirectories)
  }
  // const urls = await getUrls()
  // const urls = await debouncedGetUrls(subfolder)

  const cont = makeElement('div.mtb_sidebar')

  contextMenu = new ContextMenu(cont)
  imgGrid = makeElement('div.mtb_img_grid')
  const selector = makeSelect(
    ['input', 'output', 'video', ...output_modes, ...input_modes],
    currentMode,
  )

  selector.addEventListener('change', async (e) => {
    let newMode = e.target.value
    let changed = false
    let newSub = ''
    if (newMode !== 'input' && newMode !== 'output') {
      if (newMode.startsWith('input - ')) {
        newSub = newMode.replace('input - ', '')
        newMode = 'input'
      } else if (newMode.startsWith('output - ')) {
        newSub = newMode.replace('output - ', '')
        newMode = 'output'
      }
    }
    changed = newMode !== currentMode || newSub !== subfolder
    currentMode = newMode
    subfolder = newSub
    if (changed) {
      imgGrid.innerHTML = ''
      // const urls = await getUrls(subfolder)
      debouncedGetUrls(subfolder)
      // if (urls) {
      //   loaded_images = getImgsFromUrls(urls, imgGrid)
      // }
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
      // const urls = await getUrls(subfolder)
      // const urls = debouncedGetUrls(subfolder)
      // const urls = await getUrls(subfolder)
      debouncedGetUrls(subfolder)
      // if (urls) {
      // loaded_images = getImgsFromUrls(urls, imgGrid)
      // }
    }
  })

  const sizeSlider = makeSlider(64, 1024, currentWidth, 1)
  imgTools.appendChild(orderSelect)
  imgTools.appendChild(sizeSlider)

  loaded_images = getImgsFromUrls(urls, imgGrid)
  // infoLogger({ loaded_images })

  sizeSlider.addEventListener('input', (e) => {
    currentWidth = e.target.value
    for (const img of loaded_images) {
      img.style.width = `${e.target.value}px`
    }
  })
  handle = renderSidebar(el, cont, [selector, imgGrid, imgTools])
}

//NOTE: do not load if using the old ui
if (window?.__COMFYUI_FRONTEND_VERSION__) {
  // NOTE: removed this for now since I'm not actually exposing anything a client
  // cannot already access from "/view"...
  // let exposed = false

  const sidebar_extension = {
    name: 'mtb.io-sidebar',
    // init: async () => {
    //   try {
    //     const res = await api.fetchApi('/mtb/server-info')
    //     const msg = await res.json()
    //     exposed = msg.exposed
    //   } catch (e) {
    //     console.error('Error:', e)
    //   }
    // },
    init: () => {
      let handle
      // const version = window?.__COMFYUI_FRONTEND_VERSION__
      // console.log(`%c ${version}`, 'background: orange; color: white;')

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
        options: [
          'None',
          'Modified',
          'Modified-Reverse',
          'Name',
          'Name-Reverse',
        ],
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

          if (!loaded_images) {
            await build_ui(el)
          }
          app.api.addEventListener('status', async () => {
            if (currentMode !== 'output') return
            updateOutputsGrid()
          })
        },
        destroy: () => {
          if (handle) {
            handle.unregister()
            handle = undefined
            app.api.removeEventListener('status')
          }
        },
      })
    },
  }

  app.registerExtension(sidebar_extension)
}
