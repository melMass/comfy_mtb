/**
 * MTB API Authoring Layer
 * This file imports the compiled TypeScript bundle and registers the extension.
 *
 * The source code is in web_source/src/mtb_api/
 * Build with: cd web_source && bun run build
 */

import { registerMtbApiExtension } from './dist/mtb_api.js'

// Register the extension with ComfyUI
registerMtbApiExtension()
