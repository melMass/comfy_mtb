/**
 * File: imageFeed.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

// forked from pysssss's imageFeed.js

import { api } from '../../scripts/api.js'
import { app } from '../../scripts/app.js'
import { LocalStorageManager } from './comfy_shared.js'
const styles = {
	lighbox: {
		position: 'fixed',
		top: 0,
		left: 0,
		width: '100vw',
		height: '100vh',
		background: 'rgba(0,0,0,0.5)',
		display: 'none',
		justifyContent: 'center',
		alignItems: 'center',
		zIndex: 999,
	},
	lightboxBtn: (extra) => ({
		position: 'absolute',
		top: '50%',
		background: 'none',
		border: 'none',
		color: '#fff',
		zIndex: 1000,
		fontSize: '30px',
		cursor: 'pointer',
		pointerEvents: 'auto',
		...extra,
	}),
	img_list: {
		minHeight: '30px',
		maxHeight: '300px',
		width: '100vw',
		position: 'absolute',
		bottom: 0,
		zIndex: 10,
		background: '#333',
		overflow: 'auto',
	},
}

let currentImageIndex = 0
const imageUrls = []

let image_menu = null
const storage = new LocalStorageManager('mtb')

let activated = storage.get('image_feed', false)

app.registerExtension({
	name: 'mtb.ImageFeed',
	setup: () => {
		app.ui.settings.addSetting({
			id: 'mtb.Main.image-feed-enabled',
			category: ['mtb', 'Main', 'image-feed-enabled'],
			name: 'Enable Image Feed',
			type: 'boolean',
			defaultValue: false,
			attrs: {
				style: {
					fontFamily: 'monospace',
				},
			},
			async onChange(value) {
				storage.set('image_feed', value)
				activated = value
			},
		})
	},
	init: async () => {
		if (!activated) {
			return
		}
		const pythongossFeed = app.extensions.find(
			(e) => e.name === 'pysssss.ImageFeed',
		)
		if (pythongossFeed) {
			console.warn(
				"[mtb] - Aborting the loading of mtb's imageFeed in favor of pysssss.ImageFeed",
			)
			activated = false // just in case other methods are added later on
			return
		}
		// - HTML & CSS
		//- lightbox
		const lightboxContainer = document.createElement('div')
		Object.assign(lightboxContainer.style, styles.lighbox)

		const lightboxImage = document.createElement('img')
		Object.assign(lightboxImage.style, {
			maxHeight: '100%',
			maxWidth: '100%',
			borderRadius: '5px',
		})

		// previous and next buttons
		const lightboxPrevBtn = document.createElement('button')
		const lightboxNextBtn = document.createElement('button')

		lightboxPrevBtn.textContent = '❮'
		lightboxNextBtn.textContent = '❯'

		Object.assign(lightboxPrevBtn.style, styles.lightboxBtn({ left: '0%' }))
		Object.assign(lightboxNextBtn.style, styles.lightboxBtn({ right: '0%' }))

		// close button
		const lightboxCloseBtn = document.createElement('button')
		Object.assign(
			lightboxCloseBtn.style,
			styles.lightboxBtn({ right: '0', top: '0' }),
		)
		lightboxCloseBtn.textContent = '❌'

		const lightboxButtons = document.createElement('div')
		Object.assign(lightboxButtons.style, {
			position: 'absolute',
			top: '0%',
			right: '0%',
			// transform: "translate(50%, -50%)",
			height: '100%',
			width: '100%',
			background: 'none',
			border: 'none',
			color: '#fff',
			fontSize: '30px',
			cursor: 'pointer',
			pointerEvents: 'none',
		})

		lightboxButtons.append(lightboxPrevBtn, lightboxNextBtn, lightboxCloseBtn)
		lightboxContainer.append(lightboxButtons, lightboxImage)

		//- image list
		const imageListContainer = document.createElement('div')
		Object.assign(imageListContainer.style, styles.img_list)

		const createImgListBtn = (text, style) => {
			const btn = document.createElement('button')
			btn.type = 'button'
			btn.textContent = text
			Object.assign(btn.style, {
				...style,
				border: 'none',
				color: '#fff',
				background: 'none',
				height: '20px',
				cursor: 'pointer',
				position: 'absolute',
				top: '5px',
				fontSize: '12px',
				lineHeight: '12px',
			})
			imageListContainer.append(btn)
			return btn
		}
		const showBtn = document.createElement('button')
		const closeBtn = createImgListBtn('❌', {
			width: '20px',
			textIndent: '-4px',
			right: '5px',
		})
		const loadButton = createImgListBtn('Load Session History', {
			right: '90px',
		})
		const clearButton = createImgListBtn('Clear', {
			right: '30px',
		})

		//- tools popup button
		showBtn.classList.add('comfy-settings-btn')
		Object.assign(showBtn.style, {
			right: '16px',
			cursor: 'pointer',
			display: 'none',
		})

		//- append to DOM
		document.body.append(imageListContainer)

		showBtn.textContent = '🖼'
		showBtn.onclick = () => {
			imageListContainer.style.display = 'block'
			showBtn.style.display = 'none'
		}
		document.querySelector('.comfy-settings-btn').after(showBtn)
		document.querySelector('.comfy-settings-btn').after(lightboxContainer)

		// for (const { output } of history) {
		// 	if (output?.images) {
		// 		for (const src of output.images) {
		// 			const img = document.createElement("img");
		// 			const but = document.createElement("button");

		//- callbacks
		closeBtn.onclick = () => {
			imageListContainer.style.display = 'none'
			showBtn.style.display = 'unset'
		}

		clearButton.onclick = () => {
			imageListContainer.replaceChildren(closeBtn, clearButton, loadButton)
		}

		lightboxNextBtn.onclick = () => {
			currentImageIndex = (currentImageIndex + 1) % imageUrls.length
			const imageUrl = imageUrls[currentImageIndex]
			lightboxImage.src = imageUrl
		}

		// Modify the lightboxPrevBtn onclick callback
		lightboxPrevBtn.onclick = () => {
			currentImageIndex =
				(currentImageIndex - 1 + imageUrls.length) % imageUrls.length
			const imageUrl = imageUrls[currentImageIndex]
			lightboxImage.src = imageUrl
		}

		lightboxCloseBtn.onclick = () => {
			lightboxContainer.style.display = 'none'
		}
		lightboxImage.onclick = lightboxNextBtn.onclick
		/**
		 * This is the function that creates the image buttons for the image list
		 * They are wrapped in a button so that they can be clicked and open
		 * the image in the lightbox.
		 * @param {*} src
		 */
		const createImageBtn = (src) => {
			console.debug(`making image ${src.filename}`)
			const img = document.createElement('img')
			const but = document.createElement('button')

			Object.assign(but.style, {
				height: '120px',
				width: '120px',
				border: 'none',
				padding: 0,
				margin: 0,
			})
			Object.assign(img.style, {
				width: '100%',
				height: '100%',
				objectFit: 'cover',
			})

			img.src = `/view?filename=${encodeURIComponent(src.filename)}&type=${
				src.type
			}&subfolder=${encodeURIComponent(src.subfolder)}`

			imageUrls.push(img.src)

			console.debug(img.src)

			img.onload = () => {
				but.style.width = `${120 * (img.naturalWidth / img.naturalHeight)}px`
			}

			but.onclick = () => {
				lightboxContainer.style.display = 'flex'
				// add the same image to the lightbox
				lightboxImage.src = img.src
				// lighboxContainer.replaceChildren(lightboxButtons, img);
			}

			// add right click menu
			but.addEventListener('contextmenu', (e) => {
				e.preventDefault()

				if (image_menu) {
					image_menu.remove()
				}

				image_menu = document.createElement('div')
				Object.assign(image_menu.style, {
					position: 'absolute',
					top: `${e.clientY}px`,
					left: `${e.clientX}px`,
					background: '#333',
					color: '#fff',
					padding: '5px',
					borderRadius: '5px',
					zIndex: 999,
				})
				const load_img = document.createElement('button')
				load_img.textContent = 'Load'
				load_img.onclick = () => {
					app.handleFile(img.src)
				}

				image_menu.appendChild(load_img)
				document.body.appendChild(image_menu)
			})

			but.append(img)
			imageListContainer.prepend(but)
		}

		loadButton.onclick = async () => {
			const all_history = await api.getHistory()
			for (const history of all_history.History) {
				if (history.outputs) {
					for (const key of Object.keys(history.outputs)) {
						console.debug(key)
						if (history.outputs[key].images) {
							for (const im of history.outputs[key].images) {
								console.debug(im)
								createImageBtn(im)
							}
						}
					}
					// for (const src of outputs.outputs.images) {
					// 	console.debug(src)
					// 	makeImage(`${src.subfolder}/${src.filename}`)
					// }
				}
			}
		}

		///////-------

		// const all_history = await api.getHistory()
		// for (const history of all_history.History) {
		// 	if (history.outputs) {
		// 		for (const key of Object.keys(history.outputs)) {
		// 			for (const im of history.outputs[key].images) {
		// 				makeImage(im)
		// 			}
		// 		}
		// 		// for (const src of outputs.outputs.images) {
		// 		// 	console.debug(src)
		// 		// 	makeImage(`${src.subfolder}/${src.filename}`)
		// 		// }
		// 	}
		// }

		//- Hook into the API
		api.addEventListener('executed', ({ detail }) => {
			if (detail?.output?.images) {
				for (const src of detail.output.images) {
					console.debug(`Adding ${src} to image feed`)
					createImageBtn(src)
				}
			}
		})
	},
})
