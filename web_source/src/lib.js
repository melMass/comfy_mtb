// import './app.css'
// import App from './App.svelte'
import Inspector from './lib/Inspector.svelte'

export const createOutliner = (target, opts) => {
  const options = opts || {}
  const tgt = target || document.body

  const app = new Inspector({
    // target: document.getElementById('app'),
    target: tgt,
    props: options,
  })
  return app
}
