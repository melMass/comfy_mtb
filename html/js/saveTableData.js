/**
 * File: saveTableData.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

function saveTableData(identifier) {
  const table = document.querySelector(
    `#style-editor table[data-id='${identifier}']`
  )

  let currentData = []
  const rows = table.querySelectorAll('tr')
  const filename = table.getAttribute('data-id')

  rows.forEach((row, rowIndex) => {
    const rowData = []
    const cells =
      rowIndex === 0
        ? row.querySelectorAll('th')
        : row.querySelectorAll('td input, td textarea')

    cells.forEach((cell) => {
      rowData.push(rowIndex === 0 ? cell.textContent : cell.value)
    })

    currentData.push(rowData)
  })

  let tablesData = {}
  tablesData[filename] = currentData

  console.debug('Sending styles to manage endpoint:', tablesData)
  fetch('/mtb/actions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: 'saveStyle',
      args: tablesData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.debug('Success:', data)
    })
    .catch((error) => {
      console.error('Error:', error)
    })
}
