import { app } from '../../scripts/app.js'
import * as shared from './comfy_shared.js'

import { errorLogger, infoLogger } from './comfy_shared.js'
import * as mtb_ui from './mtb_ui.js'

const defaultOptions = {
  capabilities: {
    execute: false,
  },
  lint: false,
  mode: 'python',
  theme: 'dracula',
}

// wrapper around ACE
export class MtbEditor {
  setOptions(options) {
    this.options = shared.deepMerge(defaultOptions, options || {})
    if (
      this.options.capabilities.execute === false &&
      this.options.lint === true
    ) {
      infoLogger('ME: Disabling lint because execute is disabled')
      this.options.lint = false
    }

    if (this.options.lint) {
      this.debouncedLint = shared.debounce(this.lintCode.bind(this), 500)
    }
  }
  constructor(node, options) {
    infoLogger('ME: construct')

    this.container = mtb_ui.makeElement('div', {
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'monospace',
      height: '100%',
      padding: '5px',
      width: '100%',
    })

    this.uuid = shared.makeUUID()
    this._code = ''

    // sanitize options
    this.setOptions(options)
    this.setupUI()

    infoLogger('ME: inputDiv', this.inputDiv)

    if (node) {
      infoLogger('ME: Hooking up parent node', shared.safe_json(node))
      this.parentNode = node
      const editor = this
      infoLogger('ME: NODE', shared.safe_json(this.parentNode))

      infoLogger(
        'ME: NODE KEYS',
        shared.safe_json(Object.keys(this.parentNode).sort()),
      )


      shared.chainCallback(this.parentNode, 'onExecuted', function (res) {
        infoLogger('Executed', res)
        const { error, output_html } = res
        if (error[0]) {
          editor.appendOutput(
            `<div style="color: #f00; font-weight: bold;">Error:</div>${output_html[0]}`,
          )
        } else {
          editor.appendOutput(output_html[0])
        }
      })

      shared.chainCallback(this.parentNode, 'onConfigure', function (serial) {
        shared
          .infoLogger('ME Node: configure called', { node: this, serial })
          .notify()

        // editor.setCode(serial.properties.inputCode)
        // if (serial.properties.uuid) {
        //   editor.uuid = serial.properties.uuid
        // }

      })
      shared.chainCallback(this.parentNode, 'onRemoved', function (...serial) {
        errorLogger('ME Node: removed called TODO', {
          node: this,
          serial,
        })
      })
      shared.chainCallback(this.parentNode, 'onSerialize', function (serial) {
        infoLogger('ME Node: serialize called', {
          editor,
          node: this,
          serial,
        })

        // if (editor.aceEditor) {
        // serial.properties.inputCode = editor.aceEditor.getValue()
        // }

        const uuid_widget = this.widgets.find((e) => e.name === 'uuid')
        if (uuid_widget) {
          if (uuid_widget.value) {
            editor.uuid = uuid_widget.value
          } else {
            uuid_widget.value = editor.uuid
          }
        }

        // serial.properties.inputHeightRatio = this.properties.inputHeightRatio
      })

      // shared.chainCallback(this.parentNode,"onResize", function(size) {
      //   editor.container.style.width = `${size[0] - 10}px` // Account for padding
      //   container.style.height = `${size[1] - 10}px`
      //
      // })

      infoLogger('ME: callback setup on node')
      this.debouncedUpdateNodeProperty = shared.debounce(() => {
        if (this.aceEditor && this.parentNode) {
          const currentCode = this.aceEditor.getValue()
          // this.parentNode.setProperty('inputCode', currentCode)
          this._widget.value = currentCode
          infoLogger("ME: Node property 'inputCode' updated.")
        }
      }, 300)
    }
    this.loadAceEditor()
  }

  setupUI() {
    this.inputDiv = mtb_ui.makeElement(
      'div',
      {
        // backgroundColor: '#333',
        // color: '#eee',
        // border: '1px solid #555',
        // borderRadius: '4px',
        // marginBottom: '5px',
        boxSizing: 'border-box',
        minHeight: '100px',
        width: 'calc(100% - 10px)',
        // overflow: 'hidden',
      },
      this.container,
    )

    if (this.options.capabilities.execute) {
      this.handleDiv = mtb_ui.makeElement(
        'div',
        {
          backgroundColor: '#666',
          borderRadius: '2px',
          cursor: 'ns-resize',
          height: '5px',
          marginBottom: '5px',
          width: '100%',
        },
        this.container,
      )
      this.handleDiv.addEventListener(
        'mousedown',
        this.startResizing.bind(this),
      )

      const runButton = mtb_ui.makeElement(
        'button',
        {
          backgroundColor: '#555',
          border: 'none',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '14px',
          marginBottom: '5px',
          padding: '8px',
          width: '100%',
        },
        this.container,
      )
      runButton.textContent = 'Run Code (Ctrl+Enter)'
      runButton.onclick = () => {
        // node.__repl('RUN') //node.executeCode()
        this.executeCode()
      }
      const clearButton = mtb_ui.makeElement(
        'button',
        {
          backgroundColor: '#555',
          border: 'none',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '14px',
          marginBottom: '5px',
          padding: '8px',
          width: '100%',
        },
        this.container,
      )
      clearButton.textContent = 'Clear Output'
      clearButton.onclick = () => {
        this.outputArea.innerHTML = ''
        // this.properties.outputHistory = ''
        this.parentNode?.setProperty('outputHistory', '')
      }

      this.outputArea = mtb_ui.makeElement(
        'div',
        {
          backgroundColor: '#222',
          border: '1px solid #555',
          borderRadius: '4px',
          boxSizing: 'border-box',
          color: '#ddd',
          flexGrow: '1',
          fontFamily: 'monospace',
          fontSize: '14px',
          overflowY: 'auto',
          padding: '5px',
          whiteSpace: 'pre-wrap',
          width: 'calc(100% - 10px)',
        },
        this.container,
      )

      // augment the node
      // node.__repl = function (msg) {
      //   console.log('Called msg on api REPl', { msg, node: this })
      // }
      // node.setProperty('inputCode', '')
      // node.title = '🐍 REPL (mtb)'
    }
  }
  get code() {
    if (this._code) {
      return this._code
    }
    if (this.aceEditor) {
      return this.aceEditor.getValue()
    }
    // if (this.parentNode) {
    // return this.parentNode.properties.inputCode
    // }
    return ''
  }
  async lintCode() {
    if (this.options.mode !== 'python') {
      errorLogger('ME: Only python supports lint').notify()
      return
    }
    if (!this.aceEditor) {
      shared.warnLogger('ME: Ace Editor not loaded yet')
      return
    }
    const code = this.aceEditor.getValue()
    if (!code.trim()) {
      this.aceEditor.session.setAnnotations([])
      return
    }

    try {
      const response = await fetch('/mtb/lint', {
        body: JSON.stringify({ code: code, name: this.uuid }),
        headers: {
          'Content-Type': 'application/json',
        },
        method: 'POST',
      })

      if (!response.ok) {
        errorLogger('ME: Failed to lint', response)
        throw new Error(
          `HTTP error! status: ${response.status} ${response.statusText}`,
        )
      }

      const result = await response.json()
      // result.diagnostics should be an array of {row, column, text, type}
      this.aceEditor.session.setAnnotations(result.diagnostics)
    } catch (e) {
      errorLogger('ME: Linting Error', e)
      this.aceEditor.session.setAnnotations([
        {
          column: 0,
          row: 0,
          text: `Linting failed: ${e.message}`,
          type: 'error',
        },
      ])
    }
  }

  setupWidget(name, typeName) {
    if (this._widget) {
      throw Error('Widget already setup, this should not happen')
    }

    this._widget = this.parentNode.addDOMWidget(
      name,
      typeName,
      this.container,
      {
        getValue: () => {
          //   'Called get value of Code Editor',
          //   node.properties,
          //   editor.code,
          // )
          // return 'foobar'
          // return node.properties.inputCode
          // return this.parentNode?.properties?.inputCode // editor.code
          return this._code
        },
        hideOnZoom: false,
        setValue: (v) => {
          infoLogger('Called set value of Code Editor with:', v)
          // this.parentNode.setProperty('inputCode', v)
          this._code = v
        },
      },
    )
    return this._widget
  }

  // --- Resizing Logic ---
  startResizing(e) {
    if (!this.inputDiv) {
      infoLogger("The input div isn't ready", this)
      errorLogger("The input div isn't ready")
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
    if (!this.aceEditor || !this.isResizing) return

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
      this.parentNode.setProperty(
        'inputHeightRatio',
        new_input_height / totalDynamicHeight,
      )
    }

    this.aceEditor.resize() // Important for Ace to redraw
  }

  stopResizing() {
    this.isResizing = false
    document.removeEventListener('mousemove', this.doResize)
    document.removeEventListener('mouseup', this.stopResizing)
    document.body.style.cursor = '' // Restore default cursor
  }

  setCode(code) {
    infoLogger('called set code')
    this._code = code || this._code
    if (!this._code) {
      return
    }
    if (this._widget) {
      this._widget.value = this._code
    }

    if (this.reference) {
      this.reference.value = this._code
    }
    if (this.aceEditor) {
      this.aceEditor.setValue(this._code)
    }
    // if (this.parentNode) {
    // this.parentNode.setProperty('inputCode', this._code)
    // }
    // TODO: probably shouldn't be auto called depending on the call chain..
    if (this.options.lint) {
      this.debouncedLint()
    }
  }

  loadAceEditor() {
    if (window.MTB?.ace_loaded) {
      if (this.aceEditor) {
        infoLogger('ACE is already loaded, nothing to do')
      } else {
        this.initAceEditor()
      }
      return
    }
    // NOTE: this need more testing, globals can come from other extensions...
    //
    // let NEED_PATCH = false
    if (window.ace) {
      infoLogger(
        'A global ace was found in scope, to avoid issues with it we will patch it',
      )
      // NEED_PATCH = true
      // // window._backupAce = window.ace
      // // window.ace = null
    }

    shared
      .loadScript('/mtb_async/ace/ace.js')
      .then((m) => {
        infoLogger('ACE was loaded', m)
        // window.MTB_ACE = window.ace
        if (!window.MTB) {
          // might happen now that we don't control the lifecycle
          window.MTB = {}
        }

        window.MTB.ace_loaded = true
        this.initAceEditor()
        // this.aceEditor.setValue(this.properties.inputCode, -1)
        // this.aceEditor.setValue(this.properties.inputCode, -1)
        this.setCode()
      })
      .catch((e) => {
        errorLogger(`Error loading ace: ${e}`)
      })
      .finally(() => {
        // if (NEED_PATCH) {
        //   console.log('Patching back window object')
        //   window.ace = window._backupAce
        // }
      })
  }

  appendOutput(html) {
    this.outputArea.innerHTML += html
    this.outputArea.scrollTop = this.outputArea.scrollHeight

    if (this.parentNode) {
      const currentHistory = this.parentNode.properties.outputHistory || ''
      this.parentNode.setProperty('outputHistory', currentHistory + html)
    }
  }

  async executeCode() {
    if (!this.aceEditor) {
      infoLogger('Editor is gone, recreating it.')
      this.initAceEditor()
    }

    const code = this.aceEditor.getValue()
    if (!code.trim()) {
      infoLogger('No code to execute')
      return
    }

    const inputPrompt = `<div style="color:#888; margin-top: 10px;">>>> ${code}</div>`

    this.appendOutput(inputPrompt)

    try {
      const response = await fetch('/mtb/execute', {
        body: JSON.stringify({ code: code, name: this.uuid, reset: true }),
        headers: {
          'Content-Type': 'application/json',
        },
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      infoLogger('Received from backend', result)
      const outputHtml = result.output_html || ''
      const error = result.error

      if (error) {
        this.appendOutput(
          `<div style="color: #f00; font-weight: bold;">Error:</div>${outputHtml}`,
        )
      } else {
        this.appendOutput(outputHtml)
      }
    } catch (e) {
      const errorMessage = `<div style="color: #f00;">Frontend Error: ${e.message}</div>`

      this.appendOutput(errorMessage)
      errorLogger('MTBEditor Backend Error:', e).notify()
    } finally {
      // Not clearing
      // this.inputArea.value = '' // Clear input after execution
      // this.properties.inputCode = '' // Clear persisted input
    }
  }

  initAceEditor() {
    if (!window.MTB?.ace_loaded) {
      errorLogger('ACE editor not loaded. Cannot set up editors.')
      return
    }

    if (!this.inputDiv) {
      errorLogger('Input div not found for Ace editor initialization.')
      return
    }
    this.aceEditor = ace.edit(this.inputDiv)
    this.aceEditor.setTheme(`ace/theme/${this.options.theme || 'dracula'}`) //"ace/theme/monokai", "ace/theme/github"
    this.aceEditor.session.setMode(`ace/mode/${this.options.mode}`)
    this.aceEditor.setOptions({
      autoScrollEditorIntoView: true,
      behavioursEnabled: true,
      cursorStyle: 'ace', // "ace" | "slim" | "smooth" | "wide"
      displayIndentGuides: true,
      fixedWidthGutter: true,
      fontFamily: 'monospace',
      // enableBasicAutocompletion: true,
      // enableLiveAutocompletion: true,
      // enableSnippets: true,
      fontSize: '14px',
      hasCssTransforms: true,
      highlightActiveLine: true,
      highlightSelectedWord: true,
      scrollPastEnd: 0.5,
      showPrintMargin: false,
      tabSize: 4,
      useSoftTabs: true,
      wrap: true,
    })

    // Custom keybinding for Ctrl+Enter
    // this.aceEditor.commands.addCommand({
    //   name: 'runCode',
    //   bindKey: { win: 'Ctrl-Enter', mac: 'Command-Enter' },
    //   exec: () => this.executeCode(),
    // })

    // this.addCommand = this.aceEditor.commands.addCommand

    if (this.options.capabilities.execute) {
      this.aceEditor.commands.addCommand({
        bindKey: { mac: 'Command-Enter', win: 'Ctrl-Enter' },
        exec: () => this.executeCode(),
        name: 'runCode',
      })
    }

    // Listen for changes to trigger linting
    // for now keeping it like that but we might just not register
    // the event at all.
    this.aceEditor.session.on('change', () => {
      this.debouncedUpdateNodeProperty()
      if (this.options.lint) {
        this.debouncedLint()
      }
    })

    infoLogger('ACE editor created', { aceEditor: this.aceEditor })
    // this.outputArea.scrollTop = this.outputArea.scrollHeight
  }
}

// the widget
// This is a bit hacky and backward but the simplest I could
// find for the bidirectionality
export const CODE_EDITOR = (node, name, inputData, _app) => {
  infoLogger('CODE EDITOR NODE NOW', {
    inputData: shared.safe_json(inputData),
    node: shared.safe_json(node),
  })

  const [typeName, options] = inputData

  infoLogger('CODE_EDITOR', { name, node, options })
  switch (
    options.lang //(node.type || node.title) {
  ) {
    case undefined: {
      shared
        .errorLogger(
          'Using CODE_EDITOR without specifying lang is not supported anymore',
        )
        .notify()
      break
    }
    case 'python': {

      const editor = new MtbEditor(node, {
        capabilities: {
          execute: true, // options.allow_exec
        },
        lang: 'python',
      })

      return editor.setupWidget(name, typeName)

    }
    default: {
      break
    }
  }
}

app.registerExtension({
  getCustomWidgets: () => {
    return {
      CODE_EDITOR,
    }
  },
  name: 'mtb.code_editor',
})
