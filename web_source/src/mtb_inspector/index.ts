// import './app.css'
// import App from './App.svelte'
import { mount } from 'svelte'

import Inspector from '../lib/Inspector.svelte'

export const createOutliner = (target: HTMLElement, opts: unknown) => {
  const options = opts || {}
  const tgt = target || document.body

  const app = mount(Inspector, {
    // target: document.getElementById('app'),
    target: tgt,
    props: options,
  })
  return app
}
