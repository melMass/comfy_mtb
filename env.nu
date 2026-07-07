# NOTE: This file is only use for development you can ignore it

use log.nu
use nssm.nu *
use nutils.nu [ make-id upsert-all fwd-slash backup-file ]
use os.nu [ link ]

# --- utilities ---
def get_root [ --clean] {
  if $clean {
    $env.COMFY.ROOTS.clean
  } else {
    $env.COMFY.ROOTS.main
  }
}

def --env path-add [pth] {
  $env.PATH = ($env.PATH | append ($pth | path expand))
}

def short-date [] {
  format date "%Y-%m-%d"
}

export def spawn-for [timeout: duration task: closure] {
  let input = $in
  let parent_id = job id
  let task_id = job spawn {
    $input | do $task | job send --tag (job id) $parent_id
  }
  try {
    job recv --tag $task_id --timeout $timeout
  } catch {
    job kill $task_id
    error make {
      msg: "Task timed out."
      label: {
        text: "timed out"
        span: (metadata $task).span
      }
    }
  }
}

# --- exports --
export def "comfy profile" [timeout = 60sec] {
  let to_match = "To see the GUI go to"

  pyinstrument -r html main.py ...($env.COMFY.ARGS)
  | tee -e {
    each {
      let stde = $in
      print -ne $stde
      if $to_match in $stde {
        print $"(ansi gb)Profiling Done!(ansi reset)"
        let process = (ps -l | where name =~ python | where command =~ pyinstrument | last)
        kill -f $process.pid
      }
    }
  }
  | complete
  | get stdout
  | save $"profiled_(date now | format date '%s').html"
}

export def "comfy profile-plus" [] {

  let timestamp = (date now | format date "%s")
  let log_name = $"cprofile_run_($timestamp)"
  let profiled = (python -m cProfile main.py --port 3000 --preview-method auto | tee -e { print -ne } | complete)

  let out = (
    $profiled.stdout
    | lines
    # skip summary
    | skip 4
    | str join "\n"
  )
  # save result
  $out | save $"raw_($log_name).txt"

  # process
  $out
  | from ssv
  | upsert-all { into float } tottime percall cumtime
  | save $"($log_name).nuon"
}

export def restart-server [] {
  nssm restart -c comfy
}

# build the web components of mtb
export def "comfy build-web" [] {
  cd $env.COMFY.ROOTS.mtb
  if ("./web/dist" | path exists) {
    rm -rt ./web/dist
  }

  cd web_source
  ^$env.NPM_BINARY run build
  cp -r dist ../web/dist
}

# start the dev server for web components
export def "comfy dev-web" [] {
  cd $env.COMFY.ROOTS.mtb
  cd web_source
  ^$env.NPM_BINARY run dev
}

# daily check / update
export def "daily run" [] {
  let res = (comfy update --rebase)
  comfy update --clean
  comfy update_extensions

  daily commit $res.from_commit $res.to_commit
}

# was daily run today?
export def "daily was-run" [] {

  let daily = ($env.COMFY.ROOTS.mtb | path join daily.nuon)

  if ($daily | path exists) {
    let last = (open $daily | sort-by date | get date | last | short-date)
    let today = (date now | short-date)
    return ($last == $today)
  }
  return false
}

export def "daily commit" [from_commit: string to_commit: string] {
  let daily = ($env.COMFY.ROOTS.mtb | path join daily.nuon)
  let commit = [{date: (date now) from_commit: $from_commit to_commit: $to_commit}]

  let dailies = (
    if ($daily | path exists) {
      open $daily | append $commit
    } else {
      $commit
    }
  )

  $dailies | save -f $daily
  log success "Commited daily check"
}

# start the comfy server
export def "comfy start" [
  --clean
  --old-ui
  --listen
  --skip-daily (-s)
] {
  if not (daily was-run) and not $skip_daily {
    log info "Running daily checks"
    daily run
  }
  let root = (get_root --clean=$clean)
  cd $root

  log info "Running Server"

  MTB_DEBUG=true python main.py --port 3000 ...(if $old_ui { ["--front-end-version" "Comfy-Org/ComfyUI_legacy_frontend@latest"] } else { [--front-end-version Comfy-Org/ComfyUI_frontend@latest] }) --preview-method auto ...(if $listen { ["--listen"] } else { [] })
}

# update comfy itself and merge master in current branch
export def "comfy update" [
  --clean # comfy clean instance
  --rebase # Rebase instead of merge
] {
  let root = get_root --clean=$clean

  let models = $"($root)/models"
  let inputs = $"($root)/input"

  cd $root

  let branch_name = (git rev-parse --abbrev-ref HEAD | str trim)
  let current_commit = (git rev-parse HEAD | str trim)

  log info "Backing up and removing models symlinks"

  # preparing root for pull
  let pyproject = if not $clean {
    log info "Backing up the pyproject.toml..."
    let proj = (backup-file --root pyproject.toml)

    log info "Restoring the original pyproject"
    git checkout pyproject.toml
    cd $models
    # find and store all symlinks
    log info "Checking for links in models..."
    let links = (
      ls -la | where not ($it.target | is-empty) | select name target | sort-by name
    )
    log info $"Found links: ($links)"

    if not ($links | is-empty) {
      log info "Backing up the symlinks..."
      backup-file --root links.nuon
      $links | save -f links.nuon
      # remove them
      open links.nuon | each {|p| rm $p.name }
    }
    $proj
  } else {
    # just remove symlinks
    rm $models
    rm $inputs
  }

  cd $root

  log info $"Checking out to master"
  git checkout master

  log info "Fetching and pulling remote updates"
  if ($clean) {
    # from the local base repo master
    git fetch local master # $branch_name # master
    git pull local master # $branch_name # master
  } else {
    git fetch
    git pull
  }

  let new_commit = (git rev-parse HEAD | str trim)

  log info $"Back to our branch \(($branch_name)\)"
  git checkout -

  if $current_commit == $new_commit {
    log warn "No changes upstream"
  } else {
    if $rebase {
      log info "Rebasing changes"
      git rebase master
    } else {
      log info "Merging changes"
      git merge master
    }
  }

  log info "Linking back the models"

  if not $clean {
    rm pyproject.toml
    log info "Using our own pyproject..."
    cp $pyproject pyproject.toml
    cd $models
    log info "Relinking models..."
    # resymlink them
    open links.nuon | each {|p| link -a $p.target $p.name }
  } else {
    let master = (get_root)
    link ($master | path join models) $models
    link ($master | path join input) $inputs
  }

  let commit_count = (git rev-list --count $branch_name $"^origin/($branch_name)")

  log success $"Update successful \(($commit_count) new commits\)"

  return {from_commit: $current_commit to_commit: $new_commit}
}

export def "comfy toggle_extensions" [
  --clean
] {
  let root = get_root --clean=$clean
  cd $root
  cd custom_nodes
  let exts = (ls | where type in ["dir" "symlink"] | get name)
  let choices = ($exts | input list -m "choose extension to toggle")
  if ($choices | is-empty) {
    return
  }

  log info "Choices" $choices

  let filtered = $choices | wrap name | upsert enabled {|p| not ($p.name | str ends-with ".disabled") }

  log info "Filtered" $filtered
  $filtered | each {|f|
    let new_name = ($f.name | str replace ".disabled" "")

    let new_name = if $f.enabled {
      $"($new_name).disabled"
    } else {
      $new_name
    }
    log info $"Moving ($f.name) to ($new_name)"
    mv $f.name $new_name
  }
}

# git pull all extensions
export def "comfy update_extensions" [ --clean] {
  let root = get_root --clean=$clean
  cd $root
  cd custom_nodes
  git multipull . -s -q
}

# manual set version of mtb
export def "comfy-mtb set-version" [version: string] {
  # let pyproject = open pyproject.toml
  # let current_version = $pyproject.project.version
  # $pyproject | upsert project.version $version | save -f pyproject.toml
  # taplo format pyproject.toml
  sd "(__version__ = )\"(.*)\"" $"${1}\"($version)\"" __init__.py
  sd "(version = )(.*)" $"${1}\"($version)\"" pyproject.toml
  # log info $"⬆️ Bump version: ($current_version) → ($version)"
}

# -- env
export-env {
  $env.PYTHONUTF8 = 1
  $env.COMFY = {
    base_url : "https://mel-pc.tail3c8eb.ts.net"
    ARGS: [--port 3000 --preview-method auto]
    ROOTS: {
      mtb: ("." | path expand | fwd-slash)
      main: ("../.." | path expand | fwd-slash)
      clean: ($env.COMFY_ROOT | path dirname | path join ComfyClean | fwd-slash)
    }
  }

  $env.NPM_BINARY = "bun"
  $env.CUDA_HOME = $env.CUDA_ROOT
  #
  path-add 'C:/Portable/TensorRT-8.6.0.12/lib'
  #
  if $nu.os-info.family == 'windows' {
    path-add "G:/BIN/TensorRT-10.7.0.23/lib"
    path-add "G:/BIN/cudnn-windows-x86_64-9.6.0.74_cuda12-archive/bin"
  }
  #
  path-add ($env.CUDA_ROOT | path join bin)

  overlay use "../../.venv/Scripts/activate.nu"
}
