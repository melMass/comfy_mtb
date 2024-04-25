-- HACK: this should theorically not be needed since the lsp should read from the pyproject
-- tried: ruff-lsp or basedpyright

local comfyRoot = vim.fn.expand("%:p:h:h:h")

if not vim.env.PYTHONPATH or vim.env.PYTHONPATH == "" then
	vim.env.PYTHONPATH = comfyRoot
else
	vim.env.PYTHONPATH = vim.env.PYTHONPATH .. ";" .. comfyRoot
end
