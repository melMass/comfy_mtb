/**
 * File: foldable.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

function toggleFoldable(elementId, symbolId) {
  const content = document.getElementById(elementId)
  const symbol = document.getElementById(symbolId)
  if (content.style.display === 'none' || content.style.display === '') {
    content.style.display = 'flex'
    symbol.innerHTML = '&#9661;' // Down arrow
  } else {
    content.style.display = 'none'
    symbol.innerHTML = '&#9655;' // Right arrow
  }
}
