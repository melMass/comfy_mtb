/**
 * Adds a named stylesheet to the document with an optional ability to replace an existing one.
 *
 * @param {string} name - The unique name (ID) of the stylesheet.
 * @param {string} css - The CSS rules as a string.
 * @param {boolean} [force=false] - Whether to replace the existing stylesheet if it exists.
 * @returns {void}
 */
export function addNamedStyleSheet(name, css, force = false) {
  const existingStyleSheet = document.getElementById(name)

  if (existingStyleSheet && !force) {
    console.debug(
      `Stylesheet with name "${name}" already exists. Skipping addition.`,
    )
    return
  }

  if (existingStyleSheet && force) {
    console.debug(`Stylesheet with name "${name}" exists. Replacing...`)
    existingStyleSheet.remove()
  }

  const styleElement = document.createElement('style')
  styleElement.id = name
  styleElement.type = 'text/css'

  styleElement.appendChild(document.createTextNode(css))
  document.head.appendChild(styleElement)

  console.debug(`Stylesheet with name "${name}" added.`)
}

export const ensureMTBStyles = () => {
  const S = {
    fg: 'var(--fg-color)',
    bgi: 'var(--comfy-input-bg)',
    bgm: 'var(--comfy-menu-bg)',
    border: 'var(--comfy-border)',
    borderHover: 'var(--comfy-border-hover)',
    box: 'var(--comfy-box)',
    accent: 'var(--p-button-text-primary-color)',
  }
  const common = `
.mtb_sidebar {
  display: flex;
  flex-direction: column;
  background: ${S.bgm};
}
.mtb_img_grid {
  display: flex;
  flex-wrap: wrap;
  overflow: scroll;
  gap: 1em;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
}
.mtb_tools {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}
`
  const inputs = `
/* SELECT */
.mtb_select {
    appearance: none;
  display: grid;
  grid-template-areas: "select";
  padding: 10px;
  background-color: ${S.bgi};
  border: none;
  border-radius: 5px;
  font-size: 14px;
  color: ${S.fg};
  cursor: pointer;
  width: 100%;
}

@supports (-moz-appearance:none) {
.mtb_select{
  grid-area: select;
  background: ${S.bgi} url('data:image/gif;base64,R0lGODlhBgAGAKEDAFVVVX9/f9TU1CgmNyH5BAEKAAMALAAAAAAGAAYAAAIODA4hCDKWxlhNvmCnGwUAOw==') right center no-repeat !important;
  background-position: calc(100% - 5px) center !important;
  -moz-appearance:none !important;
}

/* styling the dropdown arrow for browsers that support it */
.mtb_select:after {
  content: "";
  width: 0.8em;
  height: 0.5em;
  background-color: ${S.fg};
  clip-path: polygon(100% 0%, 0 0%, 50% 100%);
}

.mtb_select:focus {
  outline: none;
  border-color: #0056b3;
}

.mtb_select > option {
  padding: 10px;
  background-color: ${S.bgi};
  border:none;
  color: ${S.fg};
}

.mtb_select > option:hover {
  background-color: red;
  color: ${S.fg};
}

/* SLIDER */
.mtb_slider[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 10px;
  background: ${S.bgm};
  border-radius: 5px;
  outline: none;
  opacity: 0.7;
  transition: opacity .2s;
  padding: 1em;
}

/* slider track */
.mtb_slider[type="range"]::-webkit-slider-runnable-track,
.mtb_slider[type="range"]::-moz-range-track {
  width: 100%;
  height: 10px;
  background: ${S.bgi};
  border-radius: 5px;
}


/* progress */
.mtb_slider[type="range"]::-moz-range-progress {
  background-color: ${S.accent};
  height:10px;
  border-radius: 5px;
}

/* slider thumb (the handle) */
.mtb_slider[type="range"]::-webkit-slider-thumb,
.mtb_slider[type="range"]::-moz-range-thumb
{
  -webkit-appearance: none;
  appearance: none;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: ${S.fg};
  border: none;
  cursor: pointer;
  filter: drop-shadow(1px 1px 4px black);
}

.mtb_slider[type="range"]:focus {
  opacity: 1;
}

.mtb_slider[type=range]:-moz-focusring{
    outline: 1px solid red;
    outline-offset: -1px;
}

.mtb_slider[type="range"]:hover::-webkit-slider-thumb,
.mtb_slider[type="range"]:active::-webkit-slider-thumb {
  background-color: ${S.accent};
}
`
  addNamedStyleSheet(
    'mtb_ui',
    `
${common}
${inputs}
`,
  )
}

/**
 *  Wrap an element with a div
 *
 * @param {Object} [style] - CSS styles to apply to the element.
 * @returns {HTMLElement} - The created DOM element.
 */
export const wrapElement = (element, style = {}) => {
  const container = makeElement('div', style)
  container.appendChild(element)
  return container
}

/**
 * Creates a DOM element with optional styles, class, and id.
 *
 * @param {string} kind - The tag name of the element. Supports class and id syntax (e.g. 'div.class#id').
 * @param {Object} [style] - CSS styles to apply to the element.
 * @returns {HTMLElement} - The created DOM element.
 */
export const makeElement = (kind, style) => {
  let [real_kind, className] = kind.split('.')
  let id

  if (className?.includes('#')) {
    ;[className, id] = className.split('#')
  }

  const el = document.createElement(real_kind)

  if (style) {
    Object.assign(el.style, style)
  }

  if (className) {
    el.classList.add(...className.split(' ')) // Support multiple classes
  }

  if (id) {
    el.id = id
  }

  return el
}
/**
 * Clears all child elements of the given parent element.
 *
 * @param {HTMLElement} el - The parent element whose children should be removed.
 */
export const clearElement = (el) => {
  while (el.firstChild) {
    el.removeChild(el.firstChild)
  }
}
/**
 * Creates a labeled element (input, select, etc.).
 *
 * @param {HTMLElement} el - The element to label.
 * @param {string} labelText - The label text.
 * @returns {HTMLDivElement} - A div containing the label and the element.
 */
export const makeLabeledElement = (el, labelText) => {
  const wrapper = makeElement('div.mtb_labeled_element', {
    marginBottom: '1em',
  })
  const label = makeElement('label', {
    display: 'block',
    marginBottom: '0.5em',
  })
  label.textContent = labelText
  wrapper.appendChild(label)
  wrapper.appendChild(el)
  return wrapper
}

/**
 * Converts a camelCase CSS property to kebab-case.
 *
 * @param {string} prop - The camelCase CSS property.
 * @returns {string} - The kebab-case CSS property.
 */
const camelToKebab = (prop) =>
  prop.replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`)

/**
 * Parses the style string into an object of CSS property-value pairs.
 *
 * @param {string} styleString - The CSS rule text (e.g., "color: red; background-color: blue;").
 * @returns {Object} - An object with camelCase CSS properties.
 */
const parseStyleString = (styleString) => {
  const styleObj = {}
  for (const rule of styleString.split(';')) {
    const [property, value] = rule.split(':').map((item) => item.trim())
    if (property && value) {
      const camelProp = property.replace(/-([a-z])/g, (g) => g[1].toUpperCase())
      styleObj[camelProp] = value
    }
  }
  return styleObj
}

/**
 * Defines a new CSS class with the provided styles, or skips if the class already exists.
 *
 * @param {string} className - The name of the CSS class to define.
 * @param {Object} classStyles - An object containing camelCase CSS property-value pairs.
 */
export function defineCSSClass(className, classStyles) {
  const styleSheets = document.styleSheets
  let classExists = false
  let existingStyleString = ''
  const classExistsInStyleSheet = (styleSheet) => {
    const rules = styleSheet.rules || styleSheet.cssRules
    for (const rule of rules) {
      if (rule.selectorText === `.${className}`) {
        classExists = true
        existingStyleString = rule.style.cssText // Capture existing styles
        return true
      }
    }
    return false
  }

  for (const styleSheet of styleSheets) {
    if (classExistsInStyleSheet(styleSheet)) {
      console.debug(`Class ${className} already exists, merging styles...`)
      break
    }
  }
  const existingStyles = classExists
    ? parseStyleString(existingStyleString)
    : {}
  const mergedStyles = { ...existingStyles, ...classStyles }

  const stylesString = Object.entries(mergedStyles)
    .map(([key, value]) => `${camelToKebab(key)}: ${value};`)
    .join(' ')

  if (!classExists) {
    console.debug(`Defining new class ${className}...`)
    if (styleSheets[0].insertRule) {
      styleSheets[0].insertRule(`.${className} { ${stylesString} }`, 0)
    } else if (styleSheets[0].addRule) {
      styleSheets[0].addRule(`.${className}`, stylesString, 0)
    }
  } else {
    console.debug(`Updating existing class ${className} with merged styles...`)
    for (const styleSheet of styleSheets) {
      const rules = styleSheet.rules || styleSheet.cssRules
      for (const rule of rules) {
        if (rule.selectorText === `.${className}`) {
          rule.style.cssText = stylesString // Update the existing rule
        }
      }
    }
  }

  console.debug(
    `Class ${className} has been defined/updated with styles:`,
    mergedStyles,
  )
}

/**
 * Renders a sidebar and ensures it resizes correctly when the window is resized.
 *
 * @param {HTMLElement} el - The element where the sidebar is rendered.
 * @param {HTMLElement} cont - The content container of the sidebar.
 * @param {HTMLElement[]} elems - Array of elements to append to the sidebar.
 * @returns {Object} - A handle with a method to unregister the resize event.
 */
export const renderSidebar = (el, cont, elems) => {
  el.appendChild(cont)

  if (!el.parentNode) {
    return
  }
  el.parentNode.style.overflowY = 'clip'
  cont.style.height = `${el.parentNode.offsetHeight}px`

  const resizeHandler = () => {
    cont.style.height = `${el.parentNode.offsetHeight}px`
  }
  window.addEventListener('resize', resizeHandler)

  for (const elem of elems) {
    cont.appendChild(elem)
  }

  return {
    unregister: () => {
      window.removeEventListener('resize', resizeHandler)
    },
  }
}

/**
 * Creates a <select> dropdown with given options.
 *
 * @param {string[]} options - The options for the select element.
 * @param {string} [current] - The currently selected option (optional).
 * @returns {HTMLSelectElement} - The created <select> element.
 */
export const makeSelect = (options, current = undefined) => {
  const selector = makeElement('select.mtb_select', {
    width: 'auto',
    margin: '1em',
  })

  for (const option of options) {
    const opt = makeElement('option')
    opt.value = option
    opt.innerHTML = option
    selector.appendChild(opt)
  }

  if (current !== undefined) {
    if (options.includes(current)) {
      selector.value = current
    } else {
      console.error(
        `You tried to select an option that doesn't exist (${current}). Options: ${options}`,
      )
    }
  }

  return selector
}

/**
 * Creates an <input type="range"> slider element with given parameters.
 *
 * @param {number} min - Minimum value of the slider.
 * @param {number} max - Maximum value of the slider.
 * @param {number} [value] - Initial value of the slider.
 * @param {number} [step] - Step value for the slider.
 * @returns {HTMLInputElement} - The created slider element.
 */
export const makeSlider = (min, max, value = undefined, step = undefined) => {
  const slider = makeElement('input.mtb_slider', {
    width: '100%',
  })

  slider.type = 'range'
  slider.min = min || 0
  slider.max = max || 100
  slider.value = value || slider.min
  slider.step = step || 1

  return slider
}

/**
 * Creates a button element.
 *
 * @param {string} label - The label for the button.
 * @param {Object} [style] - Optional styles to apply to the button.
 * @param {Function} [onClick] - Optional click handler.
 * @returns {HTMLButtonElement} - The created button element.
 */
export const makeButton = (label, style = {}, onClick = undefined) => {
  const button = makeElement('button.mtb_button', style)
  button.textContent = label

  if (onClick) {
    button.addEventListener('click', onClick)
  }

  return button
}

/**
 * Creates a resizable splitter between two elements.
 *
 * @param {HTMLElement} el1 - The first element.
 * @param {HTMLElement} el2 - The second element.
 * @param {'vertical' | 'horizontal'} direction - Splitter direction (vertical or horizontal).
 * @param {'absolute' | 'normal'} mode - Splitter mode: 'absolute' for free resizing, 'normal' for layout-based resizing.
 * @returns {HTMLDivElement} - The container with resizable splitter.
 */
export const makeSplitter = (
  el1,
  el2,
  direction = 'vertical',
  mode = 'normal',
) => {
  const container = makeElement('div.mtb_splitter_container', {
    display: mode === 'absolute' ? 'block' : 'flex',
    flexDirection: direction === 'vertical' ? 'row' : 'column',
    position: mode === 'absolute' ? 'relative' : 'static',
    height: '100%',
    width: '100%',
  })

  const handle = makeElement('div.mtb_splitter_handle', {
    backgroundColor: '#ccc',
    cursor: direction === 'vertical' ? 'col-resize' : 'row-resize',
    width: direction === 'vertical' ? '5px' : '100%',
    height: direction === 'horizontal' ? '5px' : '100%',
  })

  let isResizing = false

  handle.addEventListener('mousedown', () => {
    isResizing = true
  })

  window.addEventListener('mouseup', () => {
    isResizing = false
  })

  window.addEventListener('mousemove', (e) => {
    if (!isResizing) return
    if (direction === 'vertical') {
      const newWidth = e.clientX - container.offsetLeft
      el1.style.width = `${newWidth}px`
      el2.style.width = `${container.offsetWidth - newWidth}px`
    } else {
      const newHeight = e.clientY - container.offsetTop
      el1.style.height = `${newHeight}px`
      el2.style.height = `${container.offsetHeight - newHeight}px`
    }
  })

  container.appendChild(el1)
  container.appendChild(handle)
  container.appendChild(el2)

  return container
}
