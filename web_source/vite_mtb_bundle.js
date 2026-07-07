import { build } from 'vite'

/**
 * @param {{dev:boolean, styleId:string}} user_config
 * @returns {import('vite').Plugin[]} plugin
 */
const MtbBundle = (user_config) => {
  let uconfig = user_config || {}
  uconfig = { dev: false, styleId: 'mtb_style', ...uconfig }
  /** @type {import('vite').ResolvedConfig} */
  let config
  /** @type  {import('vite').Plugin[]} plugin */
  const plugins = [
    {
      apply: 'build',
      enforce: 'post',
      name: 'vite-mtb-bundle',
      config(config, env) {
        console.log(env.command)
      },
      configResolved(_config) {
        config = _config
      },
      async generateBundle(opts, bundle) {
        console.log(config.build)
        const cssAssets = Object.keys(bundle).filter(
          (i) =>
            bundle[i].type === 'asset' && bundle[i].fileName.endsWith('.css'),
        )
        const generatedStyleId =
          typeof uconfig.styleId === 'function'
            ? uconfig.styleId()
            : uconfig.styleId
      },
    },
  ]

  if (uconfig.dev) {
    plugins.push({
      name: 'vite-mtb-bundle-dev',
      apply: 'serve',
      enforce: 'post',
      transform(src, id) {
        return {
          code: src,
          map: null,
        }
      },
    })
  }

  return plugins
}

export default MtbBundle
