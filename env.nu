# NOTE: This file is only use for development you can ignore it

use private/log.nu

def get_root [--clean] {
    if $clean {
        $env.COMFY_CLEAN_ROOT
    } else {
        $env.COMFY_ROOT
    }
}

export def "comfy build-web" [] {
    cd $env.COMFY_MTB
    cd web_source
    npm run build
    cp dist/*.js ../web/dist
}

export def "comfy dev-web" [] {
    cd $env.COMFY_MTB
    cd web_source
    npm run dev
}

export def "daily run"  [] {
  let res = (comfy update --rebase)
  comfy update --clean
  comfy update_extensions

  daily commit $res.from $res.to
}

def short-date [] {
  format date "%Y-%m-%d"
}

# was daily run today?
export def "daily was-run" [] {

  let daily = ($env.COMFY_MTB | path join daily.nuon)

  if ($daily | path exists) {
    let last = (open $daily | sort-by date | get date  | last | short-date)
    let today = (date now | short-date)
    return ($last == $today)
  }
  return false
}

export def "daily commit"  [from:string, to:string] {
  let daily = ($env.COMFY_MTB | path join daily.nuon)
  let commit = [{date: (date now) from:$from to:$to}]

  let dailies = (if ($daily | path exists) {
    open $daily | append $commit
  } else {
      $commit
  })

  $dailies | save -f $daily
  log success "Commited daily check"
}

# start the comfy server
export def "comfy start" [--clean,--old-ui, --listen, --skip-daily(-s)] {
  if (not (daily was-run)) and not $skip_daily {
      log info "Running daily checks"
      daily run
    }
    let root = get_root --clean=($clean)
    cd $root

    log info "Running Server"

    MTB_DEBUG=true python main.py --port 3000 ...(if $old_ui { ["--front-end-version", "Comfy-Org/ComfyUI_legacy_frontend@latest"]} else {[ --front-end-version Comfy-Org/ComfyUI_frontend@latest]}) --preview-method auto ...(if $listen {["--listen"]} else {[]})
}

# update comfy itself and merge master in current branch
export def "comfy update" [
    --clean # ??
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
  if not $clean {
    git checkout pyproject.toml
    cd $models
    # find and store all symlinks
    let links = (ls -la |
      where not ($it.target | is-empty) |
      select name target |
      sort-by name)


    if not ($links | is-empty) {
      $links | save -f links.nuon
      # remove them
      open links.nuon | each {|p| rm $p.name }
    }
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
    cp pyproject-mel.toml pyproject.toml
    cd $models

    # resymlink them
    open links.nuon | each {|p| link -a $p.target $p.name }
  } else {
    let master = (get_root)
    link ($master | path join models) $models
    link ($master | path join input) $inputs
  }

  let commit_count = (git rev-list --count $branch_name $"^origin/($branch_name)")

  log success $"Update successful \(($commit_count) new commits\)"

  return {from:$current_commit to:$new_commit}


}

export def "comfy toggle_extensions" [--clean] {
    let root = get_root --clean=($clean)
    cd $root
    cd custom_nodes
    let exts = (ls | where type in ["dir","symlink"] | get name)
    let choices = ($exts | input list -m "choose extension to toggle")
    if ($choices | is-empty) {
        return
    }

    log info "Choices" $choices

    let filtered = $choices | wrap name | upsert enabled {|p| not ($p.name | str ends-with ".disabled")}

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
export def "comfy update_extensions" [--clean] {
    let root = get_root --clean=($clean)
    cd $root
    cd custom_nodes
    git multipull . -s -q
}

def --env path-add [pth] {
    $env.PATH = ($env.PATH | append ($pth | path expand))

}


export-env {
  $env.PYTHONUTF8 = 1
  $env.COMFY_MTB = ("." | path expand)
  # $env.CUDA_ROOT =  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\'

  $env.CUDA_HOME = $env.CUDA_ROOT

  $env.COMFY_ROOT = ("../.." | path expand)
  $env.COMFY_CLEAN_ROOT =  ($env.COMFY_ROOT | path dirname | path join ComfyClean)

  path-add 'C:/Portable/TensorRT-8.6.0.12/lib'

  if $nu.os-info.family == 'windows' {
    path-add 'G:\BIN\TensorRT-10.7.0.23\lib'
    path-add 'G:\BIN\cudnn-windows-x86_64-9.6.0.74_cuda12-archive\bin'
  }

  path-add ($env.CUDA_ROOT | path join bin)
  overlay use ../../.venv/Scripts/activate.nu
}


