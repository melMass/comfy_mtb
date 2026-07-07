const macrosPlugin = require('./mtb_macros')

module.exports = {
  // Specify the custom plugin to use
  swcPlugins: [macrosPlugin()],
}
