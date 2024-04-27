// This is a vanillajs implementation of Houdini's number input widgets.
// It basically popup a visual sensitivity slider of steps to use as incr/decr
// TODO: Convert it to IWidget

// import styles from "./style.module.css";

function getValidNumber(numberInput) {
  let num =
    isNaN(numberInput.value) || numberInput.value === ''
      ? 0
      : parseFloat(numberInput.value)
  return num
}
/**
 * Number input widgets
 */
export class NumberInputWidget {
  constructor(containerId, numberOfInputs = 1, isDebug = false) {
    this.container = document.getElementById(containerId)
    this.numberOfInputs = numberOfInputs
    this.currentInput = null // Store the currently active input

    this.threshold = 30
    this.mouseSensitivityMultiplier = 0.05
    this.debug = isDebug

    //- states
    this.initialMouseX
    this.lastMouseX
    this.activeStep = 1
    this.accumulatedDelta = 0
    this.stepLocked = false
    this.thresholdExceeded = false
    this.isDragging = false

    const styleTagId = 'mtb-constant-style'

    let styleTag = document.head.querySelector(`#${styleTagId}`)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = styleTagId

      styleTag.innerHTML = `

.${containerId}{
  margin-top: 20px;
  margin-bottom: 20px;
}
.sensitivity-menu {
  display: none;
  position: absolute;
  /* Additional styling */
}

.sensitivity-menu .step {
  cursor: pointer;
  padding: 0.5em;
  /* Add more styling as needed */
}

.sensitivity-menu {
  font-family: monospace;

  background: var(--bg-color);
  border: 1px solid var(--fg-color); 
  /* Highlight for the active step */
}
.number-input {
  background: var(--bg-color);
  color: var(--fg-color)
}

.sensitivity-menu .step.active {
  background-color:var(--drag-text); 
  /* Highlight for the active step */
}

.sensitivity-menu .step.locked {
  background-color: #f00;
  /* Change to your preferred color for the locked state */
}
#debug-container {
  transform: translateX(50%);
  width: 50%;
  text-align: center;
  font-family: monospace;
}
`
      document.head.appendChild(styleTag)
    }

    this.createWidgetElements()
    this.initializeEventListeners()
  }

  setLabel(str) {
    this.label.textContent = str
  }
  setValue(...values) {
    if (values.length !== this.numberInputs.length) {
      console.error('Number of values does not match the number of inputs.')
      console.error(
        `You provided ${values.length} but the input want ${this.numberInputs.length}`,
        { values },
      )
      return
    }
    // Set each input value
    this.numberInputs.forEach((input, index) => {
      input.value = values[index]
    })
  }
  getValue() {
    const value = []
    this.numberInputs.forEach((input, index) => {
      value.push(Number.parseFloat(input.value) || 0.0)
    })
    return value
  }
  resetValues() {
    for (const input of numberInputs) {
      input.value = 0
    }
    this.onChange?.(this.getValue())
  }

  createWidgetElements() {
    this.label = document.createElement('label')
    this.label.textContent = 'Control All:'
    this.label.className = 'widget-label'
    this.container.appendChild(this.label)

    this.label.addEventListener('mousedown', (event) => {
      if (event.button === 1) {
        this.currentInput = null
        this.handleMouseDown(event)
      }
    })

    this.label.addEventListener('contextmenu', (event) => {
      event.preventDefault()
      this.resetValues()
    })

    this.numberInputs = []

    // create linked inputs
    for (let i = 0; i < this.numberOfInputs; i++) {
      const numberInput = document.createElement('input')
      numberInput.type = 'number'
      numberInput.className = 'number-input' //styles.numberInput; //"number-input";
      numberInput.step = 'any'
      this.container.appendChild(numberInput)
      this.numberInputs.push(numberInput)

      numberInput.addEventListener('mousedown', (event) => {
        if (event.button === 1) {
          this.currentInput = numberInput
          this.handleMouseDown(event)
        }
      })
    }
    this.sensitivityMenu = document.createElement('div')
    this.sensitivityMenu.className = 'sensitivity-menu' //styles.sensitivityMenu; //"sensitivity-menu";
    this.container.appendChild(this.sensitivityMenu)

    // create steps
    const stepsValues = [0.001, 0.01, 0.1, 1, 10, 100]
    stepsValues.forEach((value) => {
      const step = document.createElement('div')
      step.className = 'step' //styles.step //"step";
      step.dataset.step = value
      step.textContent = value.toString()
      this.sensitivityMenu.appendChild(step)
    })

    this.steps = this.sensitivityMenu.getElementsByClassName('step') //styles.step)

    if (this.debug) {
      this.debugContainer = document.createElement('div')
      this.debugContainer.id = 'debug-container' //styles.debugContainer //"debugContainer";
      document.body.appendChild(this.debugContainer)
    }
  }
  showSensitivityMenu(pageX, pageY) {
    this.sensitivityMenu.style.display = 'block'
    this.sensitivityMenu.style.left = `${pageX}px`
    this.sensitivityMenu.style.top = `${pageY}px`
    this.initialMouseX = pageX
    this.lastMouseX = pageX
    this.isDragging = true
    this.thresholdExceeded = false
    this.stepLocked = false
    this.updateDebugInfo()
  }
  updateDebugInfo() {
    if (this.debug) {
      this.debugContainer.innerHTML = `
        <div>Active Step: ${this.activeStep}</div>
        <div>Initial Mouse X: ${this.initialMouseX}</div>
        <div>Last Mouse X: ${this.lastMouseX}</div>
        <div>Accumulated Delta: ${this.accumulatedDelta}</div>
        <div>Threshold Exceeded: ${this.thresholdExceeded}</div>
        <div>Step Locked: ${this.stepLocked}</div>
        <div>Number Input Value: ${this.currentInput?.value}</div>
    `
    }
  }
  handleMouseDown(event) {
    if (event.button === 1) {
      this.showSensitivityMenu(
        event.target.offsetWidth,
        event.target.offsetHeight,
      )
      event.preventDefault()
    }
  }
  handleMouseUp(event) {
    if (event.button === 1) {
      this.resetWidgetState()
    }
  }
  handleClickOutside(event) {
    if (event.target !== this.numberInput) {
      this.resetWidgetState()
    }
  }
  handleMouseMove(event) {
    if (this.sensitivityMenu.style.display === 'block') {
      const relativeY = event.pageY - 300 // this.sensitivityMenu.offsetTop

      const horizontalDistanceFromInitial = Math.abs(
        event.target.offsetWidth - this.initialMouseX,
      )

      // Unlock if the mouse moves back towards the initial position
      if (horizontalDistanceFromInitial < this.threshold) {
        this.thresholdExceeded = false
        this.stepLocked = false
        this.accumulatedDelta = 0
      }

      // Update step only if it is not locked
      if (!this.stepLocked) {
        for (let step of this.steps) {
          step.classList.remove('active') //styles.active)
          step.classList.remove('locked') //styles.locked)
          if (
            relativeY >= step.offsetTop &&
            relativeY <= step.offsetTop + step.offsetHeight
          ) {
            step.classList.add('active') //styles.active)
            this.setActiveStep(parseFloat(step.dataset.step))
          }
        }
      }

      if (this.stepLocked) {
        this.sensitivityMenu
          .querySelector('.step.active')
          ?.classList.add('locked')
      }

      this.updateStepValue(event.pageX)
    }
  }

  initializeEventListeners() {
    document.addEventListener('mousemove', (event) =>
      this.handleMouseMove(event),
    )
    document.addEventListener('mouseup', (event) => this.handleMouseUp(event))

    document.addEventListener('click', (event) =>
      this.handleClickOutside(event),
    )
  }

  setActiveStep(val) {
    if (this.activeStep !== val) {
      this.activeStep = val
      this.stepLocked = false
      this.accumulatedDelta = 0
      this.thresholdExceeded = false
    }
  }
  resetWidgetState() {
    this.sensitivityMenu.style.display = 'none'
    this.isDragging = false
    this.lastMouseX = undefined
    this.thresholdExceeded = false
    this.stepLocked = false
    this.updateDebugInfo()
  }
  updateStepValue(mouseX) {
    if (this.isDragging && this.lastMouseX !== undefined) {
      const deltaX = mouseX - this.lastMouseX
      this.accumulatedDelta += deltaX

      if (
        !this.thresholdExceeded &&
        Math.abs(this.accumulatedDelta) > this.threshold
      ) {
        this.thresholdExceeded = true
        this.stepLocked = true
      }

      if (this.thresholdExceeded && this.stepLocked) {
        // frequency of value changes
        if (
          Math.abs(this.accumulatedDelta) * this.mouseSensitivityMultiplier >=
          1
        ) {
          const valueChange = Math.sign(this.accumulatedDelta) * this.activeStep
          if (this.currentInput) {
            this.currentInput.value =
              getValidNumber(this.currentInput) + valueChange
            this.onChange?.(this.getValue())
          } else {
            this.numberInputs.forEach((input) => {
              input.value = getValidNumber(input) + valueChange
            })
          }
          this.accumulatedDelta = 0
        }
      }

      this.lastMouseX = mouseX
    }
    this.updateDebugInfo()
  }
}
