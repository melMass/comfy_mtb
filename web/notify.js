/**
 * File: notify.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

const log = (...args) => {
  if (window.MTB?.TRACE) {
    console.debug(...args)
  }
}

let transition_time = 300

const containerStyle = `
position: fixed;
top: 20px;
left: 20px;
font-family: monospace;
z-index: 99999;
height: 0;
overflow: hidden;
transition: height ${transition_time}ms ease-in-out;

`

const toastStyle = `
    background-color: #333;
    color: #fff;
    padding: 10px;
    border-radius: 5px;
    opacity: 0;
    overflow:hidden;
    height:20px;
    transition-property: opacity, height, padding;
    transition-duration: ${transition_time}ms;
  `

function notify(message, timeout = 3000) {
  log('Creating toast')
  const container = document.getElementById('mtb-notify-container')
  const toast = document.createElement('div')
  toast.style.cssText = toastStyle
  toast.innerText = message
  container.appendChild(toast)

  toast.addEventListener('transitionend', (e) => {
    // Only on out
    if (
      e.target === toast &&
      e.propertyName === 'height' &&
      e.elapsedTime > transition_time / 1000 - Number.EPSILON
    ) {
      log('Transition out')
      const totalHeight = Array.from(container.children).reduce(
        (acc, child) => acc + child.offsetHeight + 10, // Add spacing of 10px between toasts
        0
      )
      container.style.height = `${totalHeight}px`

      // If there are no toasts left, set the container's height to 0
      if (container.children.length === 0) {
        container.style.height = '0'
      }

      setTimeout(() => {
        container.removeChild(toast)
        log('Removed toast from DOM')
      }, transition_time)
    } else {
      log('Transition')
    }
  })

  // Fading in the toast
  toast.style.opacity = '1'

  // Update container's height to fit new toast
  const totalHeight = Array.from(container.children).reduce(
    (acc, child) => acc + child.offsetHeight + 10, // Add spacing of 10px between toasts
    0
  )
  container.style.height = `${totalHeight}px`

  // remove the toast after the specified timeout
  setTimeout(() => {
    // trigger the transitions
    toast.style.opacity = '0'
    toast.style.height = '0'
    toast.style.paddingTop = '0'
    toast.style.paddingBottom = '0'
  }, timeout - transition_time)
}

app.registerExtension({
  name: 'mtb.Notify',
  setup() {
    if (!window.MTB) {
      window.MTB = {}
    }

    const container = document.createElement('div')
    container.id = 'mtb-notify-container'
    container.style.cssText = containerStyle

    document.body.appendChild(container)
    window.MTB.notify = notify
    // window.MTB.notify('Hello world!')
  },
})
