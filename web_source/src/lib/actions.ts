export function dragMe(node) {
  let offsetX = 300
  let offsetY = 200
  let isDragging = false

  node.style.left = `${300}px`
  node.style.top = `${200}px`
  node.style.position = 'absolute'

  const dragPanel = (e) => {
    if (!isDragging) return
    const newX = e.clientX - offsetX
    const newY = e.clientY - offsetY
    node.style.left = `${newX}px`
    node.style.top = `${newY}px`
  }

  const startDragging = (e: MouseEvent) => {
    const target = e.target
    const isBlankSpace = target === node
    console.log({ target, node, isBlankSpace, contains: node.contains(target) })
    if (isBlankSpace) {
      isDragging = true
      offsetX = e.clientX - node.offsetLeft
      offsetY = e.clientY - node.offsetTop
      node.style.cursor = 'grabbing'
      document.addEventListener('mousemove', dragPanel)
      document.addEventListener('mouseup', stopDragging)
    }
  }

  const stopDragging = () => {
    isDragging = false
    node.style.cursor = 'grab'
    document.removeEventListener('mousemove', dragPanel)
    document.removeEventListener('mouseup', stopDragging)
  }
  // Prevent child elements from receiving mouse events during dragging
  for (const el of node.querySelectorAll('input, button')) {
    el.addEventListener('mousedown', (e) => e.stopPropagation())
  }

  node.addEventListener('mousedown', startDragging)
  return {
    destroy() {
      stopDragging()
    },
  }
}

export function resizeHandle(node) {
  let resizing = false
  let initialWidth
  let initialHeight
  let initialX
  let initialY

  const handle = document.createElement('div')
  Object.assign(handle.style, {
    position: 'absolute',
    width: '0',
    height: '0',
    bottom: '0',
    right: '0',
    cursor: 'nwse-resize',

    borderTop: '10px solid transparent',
    borderLeft: '10px solid transparent',
    borderBottom: '10px solid var(--border-color)',
    borderRight: '10px solid var(--border-color)',
    pointerEvents: 'auto',
  })

  node.appendChild(handle)

  const resizeHandler = (event) => {
    resizing = true
    initialWidth = node.offsetWidth
    initialHeight = node.offsetHeight
    initialX = event.clientX
    initialY = event.clientY

    const mouseMoveHandler = (event) => {
      if (resizing) {
        const deltaX = event.clientX - initialX
        const deltaY = event.clientY - initialY
        node.style.width = `${initialWidth + deltaX}px`
        node.style.height = `${initialHeight + deltaY}px`
        const rect = node.getBoundingClientRect()
        const newLeft = rect.left - deltaX
        const newTop = rect.top - deltaY
        node.style.left = `${newLeft}px`
        node.style.top = `${newTop}px`
      }
    }

    const mouseUpHandler = () => {
      resizing = false
      window.removeEventListener('mousemove', mouseMoveHandler)
      window.removeEventListener('mouseup', mouseUpHandler)
    }

    window.addEventListener('mousemove', mouseMoveHandler)
    window.addEventListener('mouseup', mouseUpHandler)
  }

  handle.addEventListener('mousedown', resizeHandler)

  return {
    destroy() {
      handle.removeEventListener('mousedown', resizeHandler)
      handle.remove()
    },
  }
}
