/**
 * File: splitPane.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

function initSplitPane(vertical) {
  let resizer = document.getElementById('resizer')
  let left = document.getElementById('leftPane')
  let right = document.getElementById('rightPane')
  resizer.addEventListener('mousedown', function (e) {
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', function () {
      document.removeEventListener('mousemove', onMouseMove)
    })
  })

  const onMouseMove = (e) => {
    if (vertical) {
      let leftWidth = e.clientX
      let rightWidth = window.innerWidth - e.clientX
      left.style.width = leftWidth + 'px'
      right.style.width = rightWidth + 'px'
    } else {
      let topHeight = e.clientY
      let bottomHeight = window.innerHeight - e.clientY
      left.style.height = topHeight + 'px'
      right.style.height = bottomHeight + 'px'
    }
  }
}
