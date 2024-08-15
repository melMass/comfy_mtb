# NOTE: This file is only use for development you can ignore it

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


# start the comfy server
export def "comfy start" [--clean,--old-ui, --listen] {

    let root = get_root --clean=($clean)
    cd $root
    MTB_DEBUG=true python main.py --port 3000 ...(if $old_ui { ["--front-end-version", "Comfy-Org/ComfyUI_legacy_frontend@latest"]} else {[]}) --preview-method auto ...(if $listen {["--listen"]} else {[]})
}

# update comfy itself and merge master in current branch
export def "comfy update" [
    --clean # ??
    --rebase # Rebase instead of merge
] {
    let root = get_root --clean=($clean)
    let models = $"($root)/models"
    let inputs = $"($root)/input"
    cd $root
    let branch_name = (git rev-parse --abbrev-ref HEAD | str trim)
    print $"(ansi yellow_italic)Backing up and removing models symlinks(ansi reset)"

    if not $clean {
        cd $models
        # find all symlinks
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
        rm $models
        rm $inputs
    }

    cd $root

    print $"(ansi yellow_italic)Checking out to master(ansi reset)"
    git checkout master

    print $"(ansi yellow_italic)Fetching and pulling remote updates(ansi reset)"
    git fetch
    git pull

    print $"(ansi yellow_italic)Back to our branch \(($branch_name)\)(ansi reset)"
    git checkout -

    if $rebase {
        print $"(ansi yellow_italic)Rebasing changes(ansi reset)"
        git rebase master

    } else {
        print $"(ansi yellow_italic)Merging changes(ansi reset)"
        git merge master
    }

    print $"(ansi yellow_italic)Linking back the models(ansi reset)"

    if not $clean {
        cd $models
        # resymlink them
        open links.nuon | each {|p| link -a $p.target $p.name }
    } else {
        let master = (get_root)
        link ($master | path join models) $models
        link ($master | path join input) $inputs
    }

    let commit_count = (git rev-list --count $branch_name $"^origin/($branch_name)")


    print $"(ansi green_bold)Update successful \(($commit_count) new commits\)(ansi reset)"


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

    print $choices

    let filtered = $choices | wrap name | upsert enabled {|p| not ($p.name | str ends-with ".disabled")}

    print $filtered
    $filtered | each {|f|
        let new_name = ($f.name | str replace ".disabled" "")

        let new_name = if $f.enabled {
            $"($new_name).disabled"
        } else {
            $new_name
        }
        print $"Moving ($f.name) to ($new_name)"
        mv $f.name $new_name
    }
}

# git pull all extensions
export def "comfy update_extensions" [--clean] {
    let root = get_root --clean=($clean)
    cd $root
    cd custom_nodes
    git multipull .
}

def --env path-add [pth] {
    $env.PATH = ($env.PATH | append ($pth | path expand))

}


export-env {
  $env.COMFY_MTB = ("." | path expand)
  $env.CUDA_ROOT =  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\'

  $env.CUDA_HOME = $env.CUDA_ROOT

  $env.COMFY_ROOT = ("../.." | path expand)
  $env.COMFY_CLEAN_ROOT =  ($env.COMFY_ROOT | path dirname | path join ComfyClean)

  path-add 'C:/Portable/TensorRT-8.6.0.12/lib'
  path-add ($env.CUDA_ROOT | path join bin)
  overlay use ../../.venv/Scripts/activate.nu
}


