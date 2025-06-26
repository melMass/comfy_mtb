/** Python REPL for the frontend (uses rich)*/

import { app } from '../../scripts/app.js'
import * as shared from './comfy_shared.js'
import * as mtb_ui from './mtb_ui.js'

class ComfyREPL extends LiteGraph.LGraphNode {
  constructor() {
    super()

    this.shape = LiteGraph.BOX_SHAPE
    this.isVirtualNode = true
    this.category = 'mtb/repl'
    this.title = '🐍 REPL (mtb)'

    this.uuid = shared.makeUUID()

    this.size = [600, 400]

    // Create a container for our custom widgets
    this.widget = this.addDOMWidget('HTML', 'html', this.createREPLWidget())

    this.loadAceEditor()

    // Store input and output for persistence
    this.properties = {
      inputCode: '',
      outputHistory: '',
    }

    this.outputArea.innerHTML = this.properties.outputHistory
    this.outputArea.scrollTop = this.outputArea.scrollHeight

    // Debounced linting function
    this.debouncedLint = shared.debounce(this.lintCode.bind(this), 500)

    // Resizing state variables
    this.isResizing = false
    this.initialMouseY = 0
    this.initialInputHeight = 0
    this.initialOutputHeight = 0
  }

  loadAceEditor() {
    if (window.MTB?.ace_loaded) {
      return
    }
    let NEED_PATCH = false
    if (window.ace) {
      shared.infoLogger(
        'A global ace was found in scope, to avoid issues with it we will patch it',
      )
      NEED_PATCH = true
      // window._backupAce = window.ace
      // window.ace = null
    }

    shared
      .loadScript('/mtb_async/ace/ace.js')
      .then((m) => {
        shared.infoLogger('ACE was loaded', m)
        // window.MTB_ACE = window.ace
        window.MTB.ace_loaded = true
        this.initAceEditor()

        this.aceEditor.setValue(this.properties.inputCode, -1)
      })
      .catch((e) => {
        shared.errorLogger(e)
      })
      .finally(() => {
        if (NEED_PATCH) {
          console.log('Patching back window object')
          window.ace = window._backupAce
        }
      })
  }

  initAceEditor() {
    if (!window.MTB.ace_loaded) {
      console.error('ACE editor not loaded. Cannot set up editors.')
      return
    }

    if (!this.inputDiv) {
      console.error('Input div not found for Ace editor initialization.')
      return
    }
    this.aceEditor = ace.edit(this.inputDiv)
    this.aceEditor.setTheme('ace/theme/monokai') //"ace/theme/dracula", "ace/theme/github"
    this.aceEditor.session.setMode('ace/mode/python')
    this.aceEditor.setOptions({
      enableBasicAutocompletion: true,
      enableLiveAutocompletion: true,
      enableSnippets: true,
      fontSize: '14px',
      fontFamily: 'monospace',
      showPrintMargin: false,
      wrap: true,
      tabSize: 4,
      useSoftTabs: true,
      highlightActiveLine: true,
      highlightSelectedWord: true,
      cursorStyle: 'ace', // "ace" | "slim" | "smooth" | "wide"
      behavioursEnabled: true,
      displayIndentGuides: true,
      fixedWidthGutter: true,
      scrollPastEnd: 0.5,
    })

    // Custom keybinding for Ctrl+Enter
    this.aceEditor.commands.addCommand({
      name: 'runCode',
      bindKey: { win: 'Ctrl-Enter', mac: 'Command-Enter' },
      exec: () => this.executeCode(),
    })

    // Listen for changes to trigger linting
    let lintDisabled = false
    this.aceEditor.session.on('change', () => {
      if (!lintDisabled) {
        this.debouncedLint()
      }
    })

    this.outputArea.scrollTop = this.outputArea.scrollHeight
  }

  addOutput(html) {
    this.outputArea.innerHTML += html
    this.properties.outputHistory += html
    this.outputArea.scrollTop = this.outputArea.scrollHeight
  }

  createREPLWidget() {
    const container = mtb_ui.makeElement('div', {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      boxSizing: 'border-box',
      padding: '5px',
    })

    this.inputDiv = mtb_ui.makeElement(
      'div',
      {
        width: 'calc(100% - 10px)',
        height: '100px',
        backgroundColor: '#333',
        color: '#eee',
        border: '1px solid #555',
        borderRadius: '4px',
        marginBottom: '5px',
        boxSizing: 'border-box',
        overflow: 'hidden',
      },
      container,
    )
    // Resizable Handle
    this.handleDiv = mtb_ui.makeElement(
      'div',
      {
        width: '100%',
        height: '5px',
        backgroundColor: '#666',
        cursor: 'ns-resize',
        marginBottom: '5px',
        borderRadius: '2px',
      },
      container,
    )
    this.handleDiv.addEventListener('mousedown', this.startResizing.bind(this))

    // Run Button
    this.runButton = mtb_ui.makeElement(
      'button',
      {
        width: '100%',
        padding: '8px',
        backgroundColor: '#555',
        color: '#fff',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        marginBottom: '5px',
        fontSize: '14px',
      },
      container,
    )
    this.runButton.textContent = 'Run Code (Ctrl+Enter)'
    this.runButton.onclick = () => this.executeCode()

    // Clear Button
    this.clearButton = mtb_ui.makeElement(
      'button',
      {
        width: '100%',
        padding: '8px',
        backgroundColor: '#555',
        color: '#fff',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        marginBottom: '5px',
        fontSize: '14px',
      },
      container,
    )
    this.clearButton.textContent = 'Clear Output'
    this.clearButton.onclick = () => {
      this.outputArea.innerHTML = ''
      this.properties.outputHistory = ''
    }

    // Output Area
    this.outputArea = mtb_ui.makeElement(
      'div',
      {
        flexGrow: '1',
        width: 'calc(100% - 10px)',
        backgroundColor: '#222',
        color: '#ddd',
        border: '1px solid #555',
        borderRadius: '4px',
        padding: '5px',
        fontFamily: 'monospace',
        fontSize: '14px',
        overflowY: 'auto',
        whiteSpace: 'pre-wrap',
        boxSizing: 'border-box',
      },
      container,
    )

    return container
  }

  // --- Resizing Logic ---
  startResizing(e) {
    if (!this.inputDiv) {
      shared.infoLogger("The input div isn't ready", this)
      shared.errorLogger("The input div isn't ready")
      return
    }
    this.isResizing = true
    this.initialMouseY = e.clientY
    this.initialInputHeight = this.inputDiv.offsetHeight
    this.initialOutputHeight = this.outputArea.offsetHeight

    document.addEventListener('mousemove', this.doResize.bind(this))
    document.addEventListener('mouseup', this.stopResizing.bind(this))
    document.body.style.cursor = 'ns-resize' // Change cursor globally
  }
  doResize(e) {
    if (!this.isResizing) return

    const deltaY = e.clientY - this.initialMouseY

    let new_input_height = this.initialInputHeight + deltaY
    let new_output_height = this.initialOutputHeight - deltaY

    const minInputHeight = 50 // Minimum height for Ace editor
    const minOutputHeight = 50 // Minimum height for output area

    // Clamp heights to minimums
    if (new_input_height < minInputHeight) {
      new_input_height = minInputHeight
      new_output_height =
        this.initialInputHeight + this.initialOutputHeight - minInputHeight
    }
    if (new_output_height < minOutputHeight) {
      new_output_height = minOutputHeight
      new_input_height =
        this.initialInputHeight + this.initialOutputHeight - minOutputHeight
    }

    this.inputDiv.style.height = `${new_input_height}px`
    this.outputArea.style.height = `${new_output_height}px`

    // Update the stored ratio for persistence
    const totalDynamicHeight =
      this.inputDiv.offsetHeight + this.outputArea.offsetHeight
    if (totalDynamicHeight > 0) {
      this.properties.inputHeightRatio = new_input_height / totalDynamicHeight
    }

    this.aceEditor.resize() // Important for Ace to redraw
  }

  stopResizing() {
    this.isResizing = false
    document.removeEventListener('mousemove', this.doResize)
    document.removeEventListener('mouseup', this.stopResizing)
    document.body.style.cursor = '' // Restore default cursor
  }
  // --- End Resizing Logic ---

  async executeCode() {
    const code = this.aceEditor.getValue()

    if (!code.trim()) {
      return
    }

    const inputPrompt = `<div style="color:#888; margin-top: 10px;">>>> ${code}</div>`

    this.addOutput(inputPrompt)

    try {
      const response = await fetch('/mtb/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: code, name: this.uuid }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.debug('Received from backend', result)
      const outputHtml = result.output_html || ''
      const error = result.error

      if (error) {
        this.addOutput(
          `<div style="color: #f00; font-weight: bold;">Error:</div>${outputHtml}`,
        )
      } else {
        this.addOutput(outputHtml)
      }
    } catch (e) {
      const errorMessage = `<div style="color: #f00;">Frontend Error: ${e.message}</div>`

      this.addOutput(errorMessage)
      console.error('ComfyREPL Frontend Error:', e)
    } finally {
      // Not clearing
      // this.inputArea.value = '' // Clear input after execution
      // this.properties.inputCode = '' // Clear persisted input
    }
  }

  async lintCode() {
    if (!this.aceEditor) {
      return
    }
    const code = this.aceEditor.getValue()
    if (!code.trim()) {
      this.aceEditor.session.setAnnotations([]) // Clear annotations if empty
      return
    }

    try {
      const response = await fetch('/mtb/lint', {
        // New linting endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: code, name: this.uuid }),
      })

      if (!response.ok) {
        console.log(response)
        throw new Error(
          `HTTP error! status: ${response.status} ${response.statusText}`,
        )
      }

      const result = await response.json()
      // result.diagnostics should be an array of {row, column, text, type}
      this.aceEditor.session.setAnnotations(result.diagnostics)
    } catch (e) {
      console.error('ComfyREPL Linting Error:', e)
      this.aceEditor.session.setAnnotations([
        {
          row: 0,
          column: 0,
          text: `Linting failed: ${e.message}`,
          type: 'error',
        },
      ])
    }
  }

  // Restore properties when loading a graph
  onConfigure() {
    if (this.properties.inputCode && this.aceEditor) {
      this.aceEditor.setValue(this.properties.inputCode, -1)
    }
    // if (this.properties.inputCode) {
    //   this.inputArea.value = this.properties.inputCode
    // }
    if (this.properties.outputHistory) {
      this.outputArea.innerHTML = this.properties.outputHistory
      this.outputArea.scrollTop = this.outputArea.scrollHeight
    }
    if (this.properties.uuid) {
      this.uuid = this.properties.uuid
    }
    this.debouncedLint()
    this.onResize(this.size)
  }

  // Save properties when saving a graph
  onSerialize(o) {
    if (this.aceEditor) {
      o.properties.inputCode = this.aceEditor.getValue() //this.inputArea.value
    }
    o.properties.outputHistory = this.outputArea.innerHTML
    o.properties.uuid = this.uuid
    o.properties.inputHeightRatio = this.properties.inputHeightRatio
  }

  onRemoved() {
    // Clean up DOM elements when node is removed
    if (this.widget?.element?.parentNode) {
      this.widget.element.parentNode.removeChild(this.widget.element)
    }
    // Destroy Ace editor instance to prevent memory leaks
    if (this.aceEditor) {
      this.aceEditor.destroy()
      this.aceEditor.container.remove() // Remove the Ace container div from DOM
    }
    // Clean up global event listeners if node is removed while resizing
    document.removeEventListener('mousemove', this.doResize)
    document.removeEventListener('mouseup', this.stopResizing)
    document.body.style.cursor = ''
  }
  // LiteGraph method to handle node resizing
  onResize(size) {
    // Call parent method if it exists (important for LiteGraph's internal sizing)
    if (super.onResize) {
      super.onResize(size)
    }

    // Adjust container size
    const container = this.widget.element
    container.style.width = `${size[0] - 10}px` // Account for padding
    container.style.height = `${size[1] - 10}px`

    // Adjust input and output area widths
    this.inputDiv.style.width = 'calc(100% - 10px)'
    this.outputArea.style.width = 'calc(100% - 10px)'
    //
    // const old = () => {
    //   // Calculate remaining height for output area
    //   // Ace editor manages its own height within this.inputDiv, so we use offsetHeight
    //   const inputHeight = this.inputDiv.offsetHeight
    //   const runButtonHeight = this.runButton.offsetHeight
    //   const clearButtonHeight = this.clearButton.offsetHeight
    //   const totalFixedHeight =
    //     inputHeight + runButtonHeight + clearButtonHeight + 15 // 15 for margins/padding
    //
    //   const remainingHeight = size[1] - 10 - totalFixedHeight
    //   this.outputArea.style.height = `${Math.max(50, remainingHeight)}px` // Min height 50px
    // }
    // Calculate dynamic heights
    const containerHeight = size[1] - 10
    const handleHeight = this.handleDiv.offsetHeight
    const buttonHeights =
      this.runButton.offsetHeight + this.clearButton.offsetHeight + 15 // Sum of button heights + margins

    const dynamicContentHeight = containerHeight - buttonHeights - handleHeight

    const minInputHeight = 50
    const minOutputHeight = 50

    let inputHeight = Math.max(
      minInputHeight,
      dynamicContentHeight * (this.properties.inputHeightRatio || 1.0),
    )
    let outputHeight = Math.max(
      minOutputHeight,
      dynamicContentHeight - inputHeight,
    )
    //
    // // Re-distribute if one hits its minimum
    // if (
    //   inputHeight === minInputHeight &&
    //   dynamicContentHeight - minInputHeight > minOutputHeight
    // ) {
    //   outputHeight = dynamicContentHeight - minInputHeight
    // } else if (
    //   outputHeight === minOutputHeight &&
    //   dynamicContentHeight - minOutputHeight > minInputHeight
    // ) {
    //   inputHeight = dynamicContentHeight - minOutputHeight
    // }
    //
    // // Final check to ensure total height matches available dynamic space
    // const currentTotal = inputHeight + outputHeight
    // if (currentTotal !== dynamicContentHeight) {
    //   // Adjust one of them if there's a small discrepancy due to rounding
    //   if (inputHeight > minInputHeight) {
    //     inputHeight += dynamicContentHeight - currentTotal
    //   } else if (outputHeight > minOutputHeight) {
    //     outputHeight += dynamicContentHeight - currentTotal
    //   }
    // }

    this.inputDiv.style.height = `${inputHeight}px`
    this.outputArea.style.height = `${outputHeight}px`

    // Update the ratio based on the actual heights set
    if (dynamicContentHeight > 0) {
      this.properties.inputHeightRatio = inputHeight / dynamicContentHeight
    }

    // Inform Ace editor about the resize so it can redraw its content
    if (this.aceEditor) {
      this.aceEditor.resize()
    }
  }
}

const repl = {
  name: 'mtb.repl',

  registerCustomNodes() {
    LiteGraph.registerNodeType('Python REPL', ComfyREPL)
  },
}

app.registerExtension(repl)
