/**
 * Documentation widget for nodes
 */

import { app } from '@/scripts/app'
import type { MTBNode, NodeType, NodeData, DocumentationOptions, MarkdownParser } from './types'
import { infoLogger } from './logger'

const create_documentation_stylesheet = (): void => {
  const tag = 'mtb-documentation-stylesheet'
  let styleTag = document.head.querySelector(`#${tag}`) as HTMLStyleElement | null

  if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
.documentation-popup {
    background: var(--comfy-menu-bg);
    position: absolute;
    color: var(--fg-color);
    font: 12px monospace;
    line-height: 1.5em;
    padding: 10px;
    border-radius: 6px;
    pointer-events: "inherit";
    z-index: 5;
    overflow: hidden;
}
.documentation-wrapper {
    padding: 0 2em;
    overflow: auto;
    max-height: 100%;
    &::-webkit-scrollbar {
       width: 6px;
    }
    &::-webkit-scrollbar-track {
       background: var(--bg-color);
    }
    &::-webkit-scrollbar-thumb {
       background-color: var(--fg-color);
       border-radius: 6px;
       border: 3px solid var(--bg-color);
    }
    scrollbar-width: thin;
    scrollbar-color: var(--fg-color) var(--bg-color);
    a {
      color: yellow;
    }
    a:visited {
      color: orange;
    }
    a:hover {
      color: red;
    }
}
.documentation-popup img {
  max-width: 100%;
}
.documentation-popup table {
  border-collapse: collapse;
  border: 1px var(--border-color) solid;
}
.documentation-popup th,
.documentation-popup td {
  border: 1px var(--border-color) solid;
}
.documentation-popup th {
  background-color: var(--comfy-input-bg);
}`
    document.head.appendChild(styleTag)
  }
}

let parserPromise: Promise<MarkdownParser> | undefined
const callbackQueue: Array<(parser: MarkdownParser) => void> = []

function runQueuedCallbacks(): void {
  while (callbackQueue.length) {
    const cb = callbackQueue.shift()
    if (cb && window.MTB?.mdParser) {
      cb(window.MTB.mdParser)
    }
  }
}

function loadParser(shiki: boolean): Promise<MarkdownParser> {
  if (!parserPromise) {
    parserPromise = import(
      shiki ? '/mtb_async/mtb_markdown_plus.umd.js' : '/mtb_async/mtb_markdown.umd.js'
    )
      .then(() => (shiki ? window.MTBMarkdownPlus!.getParser() : window.MTBMarkdown!.getParser()))
      .then((instance) => {
        window.MTB = window.MTB || {}
        window.MTB.mdParser = instance
        runQueuedCallbacks()
        return instance
      })
      .catch((error) => {
        // biome-ignore lint/suspicious/noConsole: error logging
        console.error('Error loading the parser:', error)
        throw error
      })
  }
  return parserPromise
}

export const ensureMarkdownParser = async (
  callback?: (parser: MarkdownParser) => void,
): Promise<MarkdownParser> => {
  infoLogger('Ensuring md parser')
  const use_shiki = app.extensionManager?.setting?.get('mtb.noteplus.use-shiki', false) as boolean

  if (window.MTB?.mdParser) {
    infoLogger('Markdown parser found')
    callback?.(window.MTB.mdParser)
    return window.MTB.mdParser
  }

  if (!parserPromise) {
    infoLogger('Running promise to fetch parser')
    try {
      loadParser(use_shiki)
    } catch (error) {
      // biome-ignore lint/suspicious/noConsole: error logging
      console.error('Error loading the parser:', error)
    }
  } else {
    infoLogger('A similar promise is already running, waiting for it to finish')
  }

  if (callback) {
    callbackQueue.push(callback)
  }

  await parserPromise

  return window.MTB!.mdParser!
}

/**
 * Add documentation widget to the given node.
 */
export const addDocumentation = (
  nodeData: NodeData,
  nodeType: NodeType,
  opts: DocumentationOptions = { icon_size: 14, icon_margin: 4 },
): void => {
  if (!nodeData.description) {
    infoLogger(`Skipping ${nodeData.name} doesn't have a description, skipping...`)
    return
  }

  const options = opts || {}
  const iconSize = options.icon_size || 14
  const iconMargin = options.icon_margin || 4

  let docElement: HTMLDivElement | null = null
  let wrapper: HTMLDivElement | null = null

  const onRem = nodeType.prototype.onRemoved

  nodeType.prototype.onRemoved = function (this: MTBNode) {
    const r = onRem ? onRem.apply(this) : undefined

    if (docElement) {
      docElement.remove()
      docElement = null
    }

    if (wrapper) {
      wrapper.remove()
      wrapper = null
    }
    return r
  }

  const drawFg = nodeType.prototype.onDrawForeground

  nodeType.prototype.onDrawForeground = function (
    this: MTBNode,
    ctx: CanvasRenderingContext2D,
    canvas: unknown,
  ) {
    const r = drawFg ? drawFg.apply(this, [ctx, canvas]) : undefined

    if (this.flags.collapsed) return r

    const x = this.size[0] - iconSize - iconMargin

    // create it
    if (this.show_doc && docElement === null) {
      create_documentation_stylesheet()

      docElement = document.createElement('div')
      docElement.classList.add('documentation-popup')
      document.body.appendChild(docElement)

      wrapper = document.createElement('div')
      wrapper.classList.add('documentation-wrapper')
      docElement.appendChild(wrapper)

      ensureMarkdownParser().then(() => {
        window.MTB!.mdParser!.parse(nodeData.description!).then((e) => {
          if (!wrapper) return
          wrapper.innerHTML = e

          // resize handle
          const resizeHandle = document.createElement('div')
          resizeHandle.classList.add('doc-resize-handle')
          Object.assign(resizeHandle.style, {
            width: '0',
            height: '0',
            position: 'absolute',
            bottom: '0',
            right: '0',
            cursor: 'se-resize',
            userSelect: 'none',
            borderWidth: '15px',
            borderStyle: 'solid',
            borderColor: 'transparent var(--border-color) var(--border-color) transparent',
          })

          wrapper.appendChild(resizeHandle)
          let isResizing = false
          let startX: number
          let startY: number
          let startWidth: number
          let startHeight: number

          resizeHandle.addEventListener(
            'mousedown',
            (e) => {
              e.stopPropagation()
              isResizing = true
              startX = e.clientX
              startY = e.clientY
              startWidth = Number.parseInt(
                document.defaultView!.getComputedStyle(docElement!).width,
                10,
              )
              startHeight = Number.parseInt(
                document.defaultView!.getComputedStyle(docElement!).height,
                10,
              )
            },
            { signal: this.docCtrl!.signal },
          )

          document.addEventListener(
            'mousemove',
            (e) => {
              if (!isResizing || !docElement) return
              const scale = app.canvas.ds.scale
              const newWidth = startWidth + (e.clientX - startX) / scale
              const newHeight = startHeight + (e.clientY - startY) / scale

              docElement.style.width = `${newWidth}px`
              docElement.style.height = `${newHeight}px`

              this.docPos = {
                width: `${newWidth}px`,
                height: `${newHeight}px`,
              }
            },
            { signal: this.docCtrl!.signal },
          )

          document.addEventListener(
            'mouseup',
            () => {
              isResizing = false
            },
            { signal: this.docCtrl!.signal },
          )
        })
      })
    } else if (!this.show_doc && docElement !== null) {
      docElement.remove()
      docElement = null
    }

    // reposition
    if (this.show_doc && docElement !== null) {
      const rect = ctx.canvas.getBoundingClientRect()
      const scaleX = rect.width / ctx.canvas.width
      const scaleY = rect.height / ctx.canvas.height
      const transform = new DOMMatrix()
        .scaleSelf(scaleX, scaleY)
        .multiplySelf(ctx.getTransform())
        .translateSelf(this.size[0] * scaleX * Math.max(1.0, window.devicePixelRatio), 0)
        .translateSelf(10, -32)

      const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)

      Object.assign(docElement.style, {
        transformOrigin: '0 0',
        transform: scale.toString(),
        left: `${transform.a + rect.x + transform.e}px`,
        top: `${transform.d + rect.y + transform.f}px`,
        width: this.docPos ? this.docPos.width : `${this.size[0] * 1.5}px`,
        height: this.docPos?.height,
      })

      if (this.docPos === undefined) {
        this.docPos = {
          width: docElement.style.width,
          height: docElement.style.height,
        }
      }
    }

    ctx.save()
    ctx.translate(x, iconSize - 34)
    ctx.scale(iconSize / 32, iconSize / 32)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 2.4
    ctx.font = 'bold 36px monospace'
    ctx.fillText('?', 0, 24)
    ctx.restore()

    return r
  }

  const mouseDown = nodeType.prototype.onMouseDown

  nodeType.prototype.onMouseDown = function (
    this: MTBNode,
    event: MouseEvent,
    localPos: [number, number],
    graphCanvas: unknown,
  ): boolean | void {
    const r = mouseDown ? mouseDown.apply(this, [event, localPos, graphCanvas]) : undefined
    const iconX = this.size[0] - iconSize - iconMargin
    const iconY = iconSize - 34

    if (
      localPos[0] > iconX &&
      localPos[0] < iconX + iconSize &&
      localPos[1] > iconY &&
      localPos[1] < iconY + iconSize
    ) {
      if (this.show_doc === undefined) {
        this.show_doc = true
      } else {
        this.show_doc = !this.show_doc
      }
      if (this.show_doc) {
        this.docCtrl = new AbortController()
      } else {
        this.docCtrl?.abort()
      }
      return true
    }

    return r
  }
}
