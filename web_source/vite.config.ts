/// <reference types="node" />
import { defineConfig, type Plugin } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js'
import path from 'node:path'

// import MtbBundle from './vite_mtb_bundle'

interface ComfyUIEnv {
  VITE_COMFYUI_HOST?: string
}
// region inlined
const rewriteComfyImports = ({
  isDev,
  comfyuiHost,
}: { isDev: boolean; comfyuiHost: string }): Plugin => {
  return {
    name: 'rewrite-comfy-imports',
    resolveId(source, importer, options) {
      // Only apply in development mode
      if (!isDev) {
        return null
      }

      // Check if the source is one of ComfyUI's core scripts
      // We'll handle both the direct '/scripts/app.js' and the aliased '@/scripts/app'
      const comfyScriptMatch = source.match(/^\/?scripts\/(app|api|ui)\.js$/)
      const comfyAliasMatch = source.match(/^@\/scripts\/(app|api|ui)$/)

      if (comfyScriptMatch || comfyAliasMatch) {
        console.log('found match')
        const scriptName = comfyScriptMatch
          ? comfyScriptMatch[1]
          : comfyAliasMatch[1]!
        // Return the full URL to the ComfyUI server's script
        return `${comfyuiHost}/scripts/${scriptName}.js`
      }

      return null
    },
  }
}
// endregion inlined

const entryPoints = {
  mtb_inspector: path.resolve(__dirname, 'src/mtb_inspector/index.ts'),
  mtb_api: path.resolve(__dirname, 'src/mtb_api/index.ts'),
  comfy_shared: path.resolve(__dirname, 'src/comfy_shared/index.ts'),
}

import noBundlePlugin from 'vite-plugin-no-bundle'
// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isDev = mode === 'development'
  const env = process.env as unknown as ComfyUIEnv
  const comfyuiHost = env.VITE_COMFYUI_HOST || 'https://mel-pc.tail3c8eb.ts.net'

  return {
    // root: path.resolve(__dirname, 'src'),
    build: {
      sourcemap: true,
      lib: {
        formats: ['es'],
        // name: 'mtb',
        // fileName: 'mtb_inspector',
        // entry: 'src/lib.js',
        entry: entryPoints,
        // fileName can be a function that receives the entry name
        // This will output files like 'mtb_inspector.js', 'mtb_react_widget.js'
        fileName: (format, entryName) => `${entryName}.js`,
      },
      rollupOptions: {
        external: [
          '/scripts/app.js',
          '/scripts/api.js',
          '/scripts/ui.js',
          '@mtb/shared',
          '/mtb_async/mtb_markdown.umd.js',
          '/mtb_async/mtb_markdown_plus.umd.js',
        ],
        // input: entryPoints,
        output: {
          chunkFileNames: 'chunks/[name]-[hash].js',
          assetFileNames: 'assets/[name][extname]',
          // Remap external imports to actual runtime paths
          paths: {
            '@mtb/shared': '../comfy_shared.js',
          },
          manualChunks(id) {
            // Create a 'vendor' chunk for all node_modules
            console.log(id)
            if (id.includes('node_modules')) {
              return 'vendor'
            }
            // You can add more specific chunking logic here if needed
            // For example, to split React and Svelte into separate vendor chunks:
            // if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
            //     return 'react-vendor';
            // }
            // if (id.includes('node_modules/svelte')) {
            //     return 'svelte-vendor';
            // }
          },
        },
      },
    },
    ssr: {
      external: ['/scripts/app.js', '/scripts/api.js', '/scripts/ui.js'],
    },
    plugins: [
      rewriteComfyImports({ isDev, comfyuiHost }),
      svelte(),
      // MtbBundle(),

      cssInjectedByJsPlugin({ styleId: 'mtb_inspector_style' }),
    ],
    resolve: {
      alias: {
        '@/scripts/app': isDev
          ? path.resolve(__dirname, 'src/mocks/app.ts')
          : '/scripts/app.js',

        '@/scripts/api': isDev
          ? path.resolve(__dirname, 'src/mocks/api.ts')
          : '/scripts/api.js',

        '@/scripts/ui': isDev
          ? path.resolve(__dirname, 'src/mocks/ui.ts')
          : '/scripts/ui.js',

        // comfy_shared.js - only alias in dev (prod uses external + output.paths)
        ...(isDev && {
          '@mtb/shared': path.resolve(__dirname, 'src/mocks/comfy_shared.ts'),
        }),
      },
    },
  }
})
