Last login: Thu Jun 29 15:33:06 on ttys001
/Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia ; exit;          
(base) shreyanshagrawal@Shreyanshs-MacBook-Air ~ % /Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia ; exit;
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.0 (2023-05-07)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> using Pkg

julia> Pkg.add("Plots")
  Installing known registries into `~/.julia`
    Updating registry at `~/.julia/registries/General.toml`
   Resolving package versions...
   Installed Preferences ────────────────── v1.4.0
   Installed JpegTurbo_jll ──────────────── v2.1.91+0
   Installed x265_jll ───────────────────── v3.5.0+0
   Installed LERC_jll ───────────────────── v3.0.0+1
   Installed Libmount_jll ───────────────── v2.35.0+0
   Installed GR_jll ─────────────────────── v0.72.7+0
   Installed libfdk_aac_jll ─────────────── v2.0.2+0
   Installed LoggingExtras ──────────────── v1.0.0
   Installed Opus_jll ───────────────────── v1.3.2+0
   Installed Xorg_xkbcomp_jll ───────────── v1.4.2+4
   Installed Measures ───────────────────── v0.3.2
   Installed RelocatableFolders ─────────── v1.0.0
   Installed ConcurrentUtilities ────────── v2.2.0
   Installed Unitful ────────────────────── v1.14.0
   Installed Xorg_xcb_util_wm_jll ───────── v0.4.1+1
   Installed Grisu ──────────────────────── v1.0.2
   Installed Xorg_xcb_util_image_jll ────── v0.4.0+1
   Installed Formatting ─────────────────── v0.4.2
   Installed PlotUtils ──────────────────── v1.3.5
   Installed Contour ────────────────────── v0.6.2
   Installed RecipesPipeline ────────────── v0.6.12
   Installed Xorg_libpthread_stubs_jll ──── v0.1.0+3
   Installed OpenSSL ────────────────────── v1.4.1
   Installed DelimitedFiles ─────────────── v1.9.1
   Installed Cairo_jll ──────────────────── v1.16.1+1
   Installed Xorg_xcb_util_jll ──────────── v0.4.0+1
   Installed Fontconfig_jll ─────────────── v2.13.93+0
   Installed Xorg_libxkbfile_jll ────────── v1.1.0+4
   Installed Libgpg_error_jll ───────────── v1.42.0+0
   Installed HTTP ───────────────────────── v1.9.8
   Installed Xorg_libXinerama_jll ───────── v1.1.4+4
   Installed ColorSchemes ───────────────── v3.21.0
   Installed IrrationalConstants ────────── v0.2.2
   Installed Showoff ────────────────────── v1.0.3
   Installed Xorg_libXau_jll ────────────── v1.0.9+4
   Installed FFMPEG ─────────────────────── v0.4.1
   Installed Missings ───────────────────── v1.1.0
   Installed Qt5Base_jll ────────────────── v5.15.3+2
   Installed JSON ───────────────────────── v0.21.4
   Installed xkbcommon_jll ──────────────── v1.4.1+0
   Installed Xorg_xcb_util_keysyms_jll ──── v0.4.0+1
   Installed Bzip2_jll ──────────────────── v1.0.8+0
   Installed Pipe ───────────────────────── v1.3.0
   Installed SimpleBufferStream ─────────── v1.1.0
   Installed HarfBuzz_jll ───────────────── v2.8.1+1
   Installed SpecialFunctions ───────────── v2.3.0
   Installed LZO_jll ────────────────────── v2.10.1+0
   Installed PlotThemes ─────────────────── v3.1.0
   Installed NaNMath ────────────────────── v1.0.2
   Installed fzf_jll ────────────────────── v0.29.0+0
   Installed FriBidi_jll ────────────────── v1.0.10+0
   Installed TranscodingStreams ─────────── v0.9.13
   Installed UnicodeFun ─────────────────── v0.4.1
   Installed x264_jll ───────────────────── v2021.5.5+0
   Installed GLFW_jll ───────────────────── v3.3.8+0
   Installed MbedTLS ────────────────────── v1.1.7
   Installed Colors ─────────────────────── v0.12.10
   Installed GR ─────────────────────────── v0.72.7
   Installed FreeType2_jll ──────────────── v2.10.4+0
   Installed JLFzf ──────────────────────── v0.1.5
   Installed StatsAPI ───────────────────── v1.6.0
   Installed DataStructures ─────────────── v0.18.14
   Installed CodecZlib ──────────────────── v0.7.1
   Installed Compat ─────────────────────── v4.7.0
   Installed Xorg_libxcb_jll ────────────── v1.13.0+3
   Installed StatsBase ──────────────────── v0.34.0
   Installed libpng_jll ─────────────────── v1.6.38+0
   Installed libaom_jll ─────────────────── v3.4.0+0
   Installed ExceptionUnwrapping ────────── v0.1.9
   Installed Scratch ────────────────────── v1.2.0
   Installed ColorTypes ─────────────────── v0.11.4
   Installed Xorg_libXext_jll ───────────── v1.3.4+4
   Installed Zstd_jll ───────────────────── v1.5.5+0
   Installed Expat_jll ──────────────────── v2.5.0+0
   Installed Libtiff_jll ────────────────── v4.4.0+0
   Installed TensorCore ─────────────────── v0.1.1
   Installed Parsers ────────────────────── v2.7.1
   Installed Plots ──────────────────────── v1.38.16
   Installed JLLWrappers ────────────────── v1.4.1
   Installed Libffi_jll ─────────────────── v3.2.2+1
   Installed ColorVectorSpace ───────────── v0.9.10
   Installed ConstructionBase ───────────── v1.5.2
   Installed Xorg_libXrender_jll ────────── v0.9.10+4
   Installed OrderedCollections ─────────── v1.6.0
   Installed Ogg_jll ────────────────────── v1.3.5+1
   Installed Xorg_libXi_jll ─────────────── v1.7.10+4
   Installed XSLT_jll ───────────────────── v1.1.34+0
   Installed Reexport ───────────────────── v1.2.2
   Installed LogExpFunctions ────────────── v0.3.24
   Installed MacroTools ─────────────────── v0.5.10
   Installed OpenSpecFun_jll ────────────── v0.5.5+0
   Installed Xorg_libXcursor_jll ────────── v1.2.0+4
   Installed Wayland_protocols_jll ──────── v1.25.0+0
   Installed Libuuid_jll ────────────────── v2.36.0+0
   Installed DocStringExtensions ────────── v0.9.3
   Installed Xorg_xcb_util_renderutil_jll ─ v0.3.9+1
   Installed Graphite2_jll ──────────────── v1.3.14+0
   Installed libass_jll ─────────────────── v0.15.1+0
   Installed Pixman_jll ─────────────────── v0.42.2+0
   Installed XML2_jll ───────────────────── v2.10.3+0
   Installed Gettext_jll ────────────────── v0.21.0+0
   Installed Wayland_jll ────────────────── v1.21.0+0
   Installed Xorg_xtrans_jll ────────────── v1.4.0+3
   Installed OpenSSL_jll ────────────────── v1.1.21+0
   Installed BitFlags ───────────────────── v0.1.7
   Installed Latexify ───────────────────── v0.16.1
   Installed Libgcrypt_jll ──────────────── v1.8.7+0
   Installed FFMPEG_jll ─────────────────── v4.4.2+2
   Installed Xorg_xkeyboard_config_jll ──── v2.27.0+4
   Installed LLVMOpenMP_jll ─────────────── v15.0.4+0
   Installed DataAPI ────────────────────── v1.15.0
   Installed Xorg_libXrandr_jll ─────────── v1.5.2+4
   Installed RecipesBase ────────────────── v1.3.4
   Installed FixedPointNumbers ──────────── v0.8.4
   Installed LAME_jll ───────────────────── v3.100.1+0
   Installed Xorg_libXfixes_jll ─────────── v5.0.3+4
   Installed PrecompileTools ────────────── v1.1.2
   Installed Libiconv_jll ───────────────── v1.16.1+2
   Installed Glib_jll ───────────────────── v2.74.0+2
   Installed LaTeXStrings ───────────────── v1.3.0
   Installed libvorbis_jll ──────────────── v1.3.7+1
   Installed URIs ───────────────────────── v1.4.2
   Installed Xorg_libX11_jll ────────────── v1.6.9+4
   Installed Libglvnd_jll ───────────────── v1.6.0+0
   Installed Requires ───────────────────── v1.3.0
   Installed Xorg_libXdmcp_jll ──────────── v1.1.3+4
   Installed SortingAlgorithms ──────────── v1.1.1
   Installed Unzip ──────────────────────── v0.2.0
   Installed UnitfulLatexify ────────────── v1.6.3
  Downloaded artifact: JpegTurbo
  Downloaded artifact: x265
  Downloaded artifact: libfdk_aac
  Downloaded artifact: GR
  Downloaded artifact: LERC
  Downloaded artifact: Opus
  Downloaded artifact: Cairo
  Downloaded artifact: Fontconfig
  Downloaded artifact: Libgpg_error
  Downloaded artifact: Qt5Base
  Downloaded artifact: Bzip2
  Downloaded artifact: HarfBuzz
  Downloaded artifact: LZO
  Downloaded artifact: fzf
  Downloaded artifact: FriBidi
  Downloaded artifact: GLFW
  Downloaded artifact: x264
  Downloaded artifact: FreeType2
  Downloaded artifact: libpng
  Downloaded artifact: libaom
  Downloaded artifact: Zstd
  Downloaded artifact: Expat
  Downloaded artifact: Libtiff
  Downloaded artifact: Libffi
  Downloaded artifact: Ogg
  Downloaded artifact: XSLT
  Downloaded artifact: OpenSpecFun
  Downloaded artifact: Wayland_protocols
  Downloaded artifact: Graphite2
  Downloaded artifact: libass
  Downloaded artifact: Pixman
  Downloaded artifact: XML2
  Downloaded artifact: Gettext
  Downloaded artifact: OpenSSL
  Downloaded artifact: FFMPEG
  Downloaded artifact: Libgcrypt
  Downloaded artifact: LLVMOpenMP
  Downloaded artifact: LAME
  Downloaded artifact: Libiconv
  Downloaded artifact: Glib
  Downloaded artifact: libvorbis
    Updating `~/.julia/environments/v1.9/Project.toml`
  [91a5bcdd] + Plots v1.38.16
    Updating `~/.julia/environments/v1.9/Manifest.toml`
  [d1d4a3ce] + BitFlags v0.1.7
  [944b1d66] + CodecZlib v0.7.1
  [35d6a980] + ColorSchemes v3.21.0
  [3da002f7] + ColorTypes v0.11.4
  [c3611d14] + ColorVectorSpace v0.9.10
  [5ae59095] + Colors v0.12.10
  [34da2185] + Compat v4.7.0
  [f0e56b4a] + ConcurrentUtilities v2.2.0
  [187b0558] + ConstructionBase v1.5.2
  [d38c429a] + Contour v0.6.2
  [9a962f9c] + DataAPI v1.15.0
  [864edb3b] + DataStructures v0.18.14
  [8bb1440f] + DelimitedFiles v1.9.1
  [ffbed154] + DocStringExtensions v0.9.3
  [460bff9d] + ExceptionUnwrapping v0.1.9
  [c87230d0] + FFMPEG v0.4.1
  [53c48c17] + FixedPointNumbers v0.8.4
  [59287772] + Formatting v0.4.2
  [28b8d3ca] + GR v0.72.7
  [42e2da0e] + Grisu v1.0.2
  [cd3eb016] + HTTP v1.9.8
  [92d709cd] + IrrationalConstants v0.2.2
  [1019f520] + JLFzf v0.1.5
  [692b3bcd] + JLLWrappers v1.4.1
  [682c06a0] + JSON v0.21.4
  [b964fa9f] + LaTeXStrings v1.3.0
  [23fbe1c1] + Latexify v0.16.1
  [2ab3a3ac] + LogExpFunctions v0.3.24
  [e6f89c97] + LoggingExtras v1.0.0
  [1914dd2f] + MacroTools v0.5.10
  [739be429] + MbedTLS v1.1.7
  [442fdcdd] + Measures v0.3.2
  [e1d29d7a] + Missings v1.1.0
  [77ba4419] + NaNMath v1.0.2
  [4d8831e6] + OpenSSL v1.4.1
  [bac558e1] + OrderedCollections v1.6.0
  [69de0a69] + Parsers v2.7.1
  [b98c9c47] + Pipe v1.3.0
  [ccf2f8ad] + PlotThemes v3.1.0
  [995b91a9] + PlotUtils v1.3.5
  [91a5bcdd] + Plots v1.38.16
  [aea7be01] + PrecompileTools v1.1.2
  [21216c6a] + Preferences v1.4.0
  [3cdcf5f2] + RecipesBase v1.3.4
  [01d81517] + RecipesPipeline v0.6.12
  [189a3867] + Reexport v1.2.2
  [05181044] + RelocatableFolders v1.0.0
  [ae029012] + Requires v1.3.0
  [6c6a2e73] + Scratch v1.2.0
  [992d4aef] + Showoff v1.0.3
  [777ac1f9] + SimpleBufferStream v1.1.0
  [a2af1166] + SortingAlgorithms v1.1.1
  [276daf66] + SpecialFunctions v2.3.0
  [82ae8749] + StatsAPI v1.6.0
  [2913bbd2] + StatsBase v0.34.0
  [62fd8b95] + TensorCore v0.1.1
  [3bb67fe8] + TranscodingStreams v0.9.13
  [5c2747f8] + URIs v1.4.2
  [1cfade01] + UnicodeFun v0.4.1
  [1986cc42] + Unitful v1.14.0
  [45397f5d] + UnitfulLatexify v1.6.3
  [41fe7b60] + Unzip v0.2.0
  [6e34b625] + Bzip2_jll v1.0.8+0
  [83423d85] + Cairo_jll v1.16.1+1
  [2e619515] + Expat_jll v2.5.0+0
  [b22a6f82] + FFMPEG_jll v4.4.2+2
  [a3f928ae] + Fontconfig_jll v2.13.93+0
  [d7e528f0] + FreeType2_jll v2.10.4+0
  [559328eb] + FriBidi_jll v1.0.10+0
  [0656b61e] + GLFW_jll v3.3.8+0
  [d2c73de3] + GR_jll v0.72.7+0
  [78b55507] + Gettext_jll v0.21.0+0
  [7746bdde] + Glib_jll v2.74.0+2
  [3b182d85] + Graphite2_jll v1.3.14+0
  [2e76f6c2] + HarfBuzz_jll v2.8.1+1
  [aacddb02] + JpegTurbo_jll v2.1.91+0
  [c1c5ebd0] + LAME_jll v3.100.1+0
  [88015f11] + LERC_jll v3.0.0+1
  [1d63c593] + LLVMOpenMP_jll v15.0.4+0
  [dd4b983a] + LZO_jll v2.10.1+0
⌅ [e9f186c6] + Libffi_jll v3.2.2+1
  [d4300ac3] + Libgcrypt_jll v1.8.7+0
  [7e76a0d4] + Libglvnd_jll v1.6.0+0
  [7add5ba3] + Libgpg_error_jll v1.42.0+0
  [94ce4f54] + Libiconv_jll v1.16.1+2
  [4b2f31a3] + Libmount_jll v2.35.0+0
⌅ [89763e89] + Libtiff_jll v4.4.0+0
  [38a345b3] + Libuuid_jll v2.36.0+0
  [e7412a2a] + Ogg_jll v1.3.5+1
⌅ [458c3c95] + OpenSSL_jll v1.1.21+0
  [efe28fd5] + OpenSpecFun_jll v0.5.5+0
  [91d4177d] + Opus_jll v1.3.2+0
  [30392449] + Pixman_jll v0.42.2+0
  [ea2cea3b] + Qt5Base_jll v5.15.3+2
  [a2964d1f] + Wayland_jll v1.21.0+0
  [2381bf8a] + Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] + XML2_jll v2.10.3+0
  [aed1982a] + XSLT_jll v1.1.34+0
  [4f6342f7] + Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] + Xorg_libXau_jll v1.0.9+4
  [935fb764] + Xorg_libXcursor_jll v1.2.0+4
  [a3789734] + Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] + Xorg_libXext_jll v1.3.4+4
  [d091e8ba] + Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] + Xorg_libXi_jll v1.7.10+4
  [d1454406] + Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] + Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] + Xorg_libXrender_jll v0.9.10+4
  [14d82f49] + Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] + Xorg_libxcb_jll v1.13.0+3
  [cc61e674] + Xorg_libxkbfile_jll v1.1.0+4
  [12413925] + Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] + Xorg_xcb_util_jll v0.4.0+1
  [975044d2] + Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] + Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] + Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] + Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] + Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] + Xorg_xtrans_jll v1.4.0+3
  [3161d3a3] + Zstd_jll v1.5.5+0
⌅ [214eeab7] + fzf_jll v0.29.0+0
  [a4ae2306] + libaom_jll v3.4.0+0
  [0ac62f75] + libass_jll v0.15.1+0
  [f638f0a6] + libfdk_aac_jll v2.0.2+0
  [b53b4c65] + libpng_jll v1.6.38+0
  [f27f6e37] + libvorbis_jll v1.3.7+1
  [1270edf5] + x264_jll v2021.5.5+0
  [dfaa095f] + x265_jll v3.5.0+0
  [d8fb68d0] + xkbcommon_jll v1.4.1+0
  [0dad84c5] + ArgTools v1.1.1
  [56f22d72] + Artifacts
  [2a0f44e3] + Base64
  [ade2ca70] + Dates
  [f43a241f] + Downloads v1.6.0
  [7b1f6079] + FileWatching
  [b77e0a4c] + InteractiveUtils
  [b27032c2] + LibCURL v0.6.3
  [76f85450] + LibGit2
  [8f399da3] + Libdl
  [37e2e46d] + LinearAlgebra
  [56ddb016] + Logging
  [d6f4376e] + Markdown
  [a63ad114] + Mmap
  [ca575930] + NetworkOptions v1.2.0
  [44cfe95a] + Pkg v1.9.0
  [de0858da] + Printf
  [3fa0cd96] + REPL
  [9a3f8284] + Random
  [ea8e919c] + SHA v0.7.0
  [9e88b42a] + Serialization
  [6462fe0b] + Sockets
  [2f01184e] + SparseArrays
  [10745b16] + Statistics v1.9.0
  [fa267f1f] + TOML v1.0.3
  [a4e569a6] + Tar v1.10.0
  [8dfed614] + Test
  [cf7118a7] + UUIDs
  [4ec0a83e] + Unicode
  [e66e0078] + CompilerSupportLibraries_jll v1.0.2+0
  [deac9b47] + LibCURL_jll v7.84.0+0
  [29816b5a] + LibSSH2_jll v1.10.2+0
  [c8ffd9c3] + MbedTLS_jll v2.28.2+0
  [14a3606d] + MozillaCACerts_jll v2022.10.11
  [4536629a] + OpenBLAS_jll v0.3.21+4
  [05823500] + OpenLibm_jll v0.8.1+0
  [efcefdf7] + PCRE2_jll v10.42.0+0
  [bea87d4a] + SuiteSparse_jll v5.10.1+6
  [83775a58] + Zlib_jll v1.2.13+0
  [8e850b90] + libblastrampoline_jll v5.7.0+0
  [8e850ede] + nghttp2_jll v1.48.0+0
  [3f19e933] + p7zip_jll v17.4.0+0
        Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`
Precompiling project...
  131 dependencies successfully precompiled in 99 seconds. 7 already precompiled.

julia> N=50
50

julia> N=50;

julia> S = range(1., N, step=0.1);

julia> L(S, o) = S*log(o) + (N-S)*log(1. - o)
L (generic function with 1 method)

julia> build PlotlyJS
ERROR: syntax: extra token "PlotlyJS" after end of expression
Stacktrace:
 [1] top-level scope
   @ none:1

julia> build PlotlyJS
ERROR: syntax: extra token "PlotlyJS" after end of expression
Stacktrace:
 [1] top-level scope
   @ none:1

(@v1.9) pkg> build PlotlyJS
ERROR: The following package names could not be resolved:
 * PlotlyJS (not found in project or manifest)

(@v1.9) pkg> build PlotlyJS
ERROR: The following package names could not be resolved:
 * PlotlyJS (not found in project or manifest)

(@v1.9) pkg> export JULIA_PKG_SERVER="https://pkg.julialang.org/registry/"

ERROR: `export` is not a recognized command. Type ? for help with available commands

(@v1.9) pkg> export JULIA_PKG_SERVER="https://pkg.julialang.org/registry/"

ERROR: `export` is not a recognized command. Type ? for help with available commands

(@v1.9) pkg> using Pkg
             Pkg.Registry.delete!(Pkg.Types.Context().env.project_path)

ERROR: `using` is not a recognized command. Type ? for help with available commands

julia> p1 = suurface(S,o, (S,o)) -> L(S,o),color=:jet, xlabel="S", ylabel="o"
ERROR: syntax: invalid assignment location ""S"" around REPL[10]:1
Stacktrace:
 [1] top-level scope
   @ REPL[10]:1

julia> 1 = suurface(S,o, (S,o)) -> L(S,o),color=:jet, xlabel="S", ylabel="o"
ERROR: syntax: invalid assignment location ""S"" around REPL[11]:1
Stacktrace:
 [1] top-level scope
   @ REPL[11]:1

julia> 1 = suurface(S,o, (S,o)) -> L(S,o),color=:jet, xlabel="S", ylabel="o"
ERROR: syntax: invalid assignment location ""S"" around REPL[11]:1
Stacktrace:
 [1] top-level scope
   @ REPL[11]:1

julia> o = range(0.1, 0.9, length=100);

julia> p1 = surface(S,o, (S,o) -> L(S,o),color=:jet, xlabel="S", ylabel="o"


ERROR: syntax: incomplete: premature end of input
Stacktrace:
 [1] top-level scope
   @ none:1

julia> 

julia> 

julia> 

julia> 

julia> Pkg.add("PlotLyJS")
ERROR: The following package names could not be resolved:
 * PlotLyJS (not found in project, manifest or registry)
   Suggestions: PlotlyJS Plotly PlotlyBase PlotlySave PlotlyLight PlotlyKaleido
Stacktrace:
  [1] pkgerror(msg::String)
    @ Pkg.Types /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:69
  [2] ensure_resolved(ctx::Pkg.Types.Context, manifest::Pkg.Types.Manifest, pkgs::Vector{Pkg.Types.PackageSpec}; registry::Bool)
    @ Pkg.Types /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:995
  [3] ensure_resolved
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:946 [inlined]
  [4] add(ctx::Pkg.Types.Context, pkgs::Vector{Pkg.Types.PackageSpec}; preserve::Pkg.Types.PreserveLevel, platform::Base.BinaryPlatforms.Platform, kwargs::Base.Pairs{Symbol, Base.TTY, Tuple{Symbol}, NamedTuple{(:io,), Tuple{Base.TTY}}})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:264
  [5] add(pkgs::Vector{Pkg.Types.PackageSpec}; io::Base.TTY, kwargs::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:156
  [6] add(pkgs::Vector{Pkg.Types.PackageSpec})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:145
  [7] #add#25
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:144 [inlined]
  [8] add
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:144 [inlined]
  [9] #add#24
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:143 [inlined]
 [10] add(pkg::String)
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:143
 [11] top-level scope
    @ REPL[14]:1

julia> gr()
ERROR: UndefVarError: `gr` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[15]:1

julia> gr()Plots.GRBackend()
ERROR: UndefVarError: `gr` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[16]:1

julia> gr()
ERROR: UndefVarError: `gr` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[17]:1

julia> p1= surface(S,o,(S,o) -> L(S,o), color =:jet , xlabel="S", ylabel="o" , title = "")
ERROR: UndefVarError: `surface` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[18]:1

julia> p1= surface(S,o,(S,o) -> L(S,o), color =:jet , xlabel="S", ylabel="o" , title = "")
ERROR: UndefVarError: `surface` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[18]:1

julia> p1= surface(S,o,(S,o) -> L(S,o), color =:jet , xlabel="S", ylabel="o" , title = "")
ERROR: UndefVarError: `surface` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[18]:1

julia> using PlotlyJS
 │ Package PlotlyJS not found, but a package named PlotlyJS is available from a
 │ registry. 
 │ Install package?
 │   (@v1.9) pkg> add PlotlyJS 
 └ (y/n/o) [y]: y   
   Resolving package versions...
   Installed AssetRegistry ─────────────── v0.1.0
   Installed Hiccup ────────────────────── v0.2.2
   Installed Lazy ──────────────────────── v0.15.1
   Installed Pidfile ───────────────────── v1.3.0
   Installed PlotlyBase ────────────────── v0.8.19
   Installed WebSockets ────────────────── v1.6.0
   Installed Blink ─────────────────────── v0.12.8
   Installed WebIO ─────────────────────── v0.8.21
   Installed Kaleido_jll ───────────────── v0.2.1+0
   Installed UnPack ────────────────────── v1.0.2
   Installed DataValueInterfaces ───────── v1.0.0
   Installed Mustache ──────────────────── v1.0.17
   Installed TableTraits ───────────────── v1.0.1
   Installed PlotlyJS ──────────────────── v0.18.10
   Installed Widgets ───────────────────── v0.6.6
   Installed JSExpr ────────────────────── v0.5.4
   Installed FunctionalCollections ─────── v0.5.0
   Installed Tables ────────────────────── v1.10.1
   Installed Mux ───────────────────────── v1.0.1
   Installed IteratorInterfaceExtensions ─ v1.0.0
   Installed Observables ───────────────── v0.5.4
   Installed Parameters ────────────────── v0.12.3
  Downloaded artifact: plotly-base-artifacts
  Downloaded artifact: web
  Downloaded artifact: electronjs_app
  Downloaded artifact: plotly-artifacts
  Downloaded artifact: Kaleido
    Updating `~/.julia/environments/v1.9/Project.toml`
  [f0f68f2c] + PlotlyJS v0.18.10
    Updating `~/.julia/environments/v1.9/Manifest.toml`
  [bf4720bc] + AssetRegistry v0.1.0
  [ad839575] + Blink v0.12.8
  [e2d170a0] + DataValueInterfaces v1.0.0
  [de31a74c] + FunctionalCollections v0.5.0
  [9fb69e20] + Hiccup v0.2.2
  [82899510] + IteratorInterfaceExtensions v1.0.0
  [97c1335a] + JSExpr v0.5.4
  [50d2b5c4] + Lazy v0.15.1
  [ffc61752] + Mustache v1.0.17
  [a975b10e] + Mux v1.0.1
  [510215fc] + Observables v0.5.4
  [d96e819e] + Parameters v0.12.3
  [fa939f87] + Pidfile v1.3.0
  [a03496cd] + PlotlyBase v0.8.19
  [f0f68f2c] + PlotlyJS v0.18.10
  [3783bdb8] + TableTraits v1.0.1
  [bd369af6] + Tables v1.10.1
  [3a884ed6] + UnPack v1.0.2
  [0f1e0344] + WebIO v0.8.21
  [104b5d7c] + WebSockets v1.6.0
  [cc8bc4a8] + Widgets v0.6.6
  [f7e6163d] + Kaleido_jll v0.2.1+0
  [8ba89e20] + Distributed
    Building WebIO ───→ `~/.julia/scratchspaces/44cfe95a-1eb2-52ea-b672-e2afdf69b78f/0eef0765186f7452e52236fa42ca8c9b3c11c6e3/build.log`
    Building PlotlyJS → `~/.julia/scratchspaces/44cfe95a-1eb2-52ea-b672-e2afdf69b78f/7452869933cd5af22f59557390674e8679ab2338/build.log`
Precompiling project...
  22 dependencies successfully precompiled in 12 seconds. 138 already precompiled.

julia> using PlotlyJS, CSV, HTTP, DataFrames
 │ Packages [CSV, HTTP, DataFrames] not found, but packages named [CSV, HTTP,
 │ DataFrames] are available from a registry. 
 │ Install packages?
 │   (@v1.9) pkg> add CSV HTTP DataFrames 
 └ (y/n/o) [y]: y
   Resolving package versions...
   Installed WeakRefStrings ───── v1.4.2
   Installed InlineStrings ────── v1.4.0
   Installed InvertedIndices ──── v1.3.0
   Installed WorkerUtilities ──── v1.6.1
   Installed SnoopPrecompile ──── v1.0.3
   Installed SentinelArrays ───── v1.4.0
   Installed PooledArrays ─────── v1.4.2
   Installed Crayons ──────────── v4.1.1
   Installed FilePathsBase ────── v0.9.20
   Installed StringManipulation ─ v0.3.0
   Installed DataFrames ───────── v1.5.0
   Installed CSV ──────────────── v0.10.11
   Installed PrettyTables ─────── v2.2.4
    Updating `~/.julia/environments/v1.9/Project.toml`
  [336ed68f] + CSV v0.10.11
  [a93c6f00] + DataFrames v1.5.0
  [cd3eb016] + HTTP v1.9.8
    Updating `~/.julia/environments/v1.9/Manifest.toml`
  [336ed68f] + CSV v0.10.11
  [a8cc5b0e] + Crayons v4.1.1
  [a93c6f00] + DataFrames v1.5.0
  [48062228] + FilePathsBase v0.9.20
  [842dd82b] + InlineStrings v1.4.0
  [41ab1584] + InvertedIndices v1.3.0
  [2dfb63ee] + PooledArrays v1.4.2
  [08abe8d2] + PrettyTables v2.2.4
  [91c51154] + SentinelArrays v1.4.0
  [66db9d55] + SnoopPrecompile v1.0.3
  [892a3eda] + StringManipulation v0.3.0
  [ea10d353] + WeakRefStrings v1.4.2
  [76eceee3] + WorkerUtilities v1.6.1
  [9fa8497b] + Future
Precompiling project...
  14 dependencies successfully precompiled in 35 seconds. 160 already precompiled.

julia> # Read data from a csv
       df =CSV.File(HTTP.get("https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv").body
       ) |> DataFrame
25×25 DataFrame
 Row │ Column1  0            1          2          3         4         5      ⋯
     │ Int64    Float64      Float64    Float64    Float64   Float64   Float6 ⋯
─────┼─────────────────────────────────────────────────────────────────────────
   1 │       0  27.8099       49.6194    83.0807   116.663   130.414   150.72 ⋯
   2 │       1  27.7197       48.5502    65.2137    95.2767  116.996   133.90
   3 │       2  30.4267       33.4775    44.8095    62.4749   77.4352  104.21
   4 │       3  16.6655       30.1086    39.9695    44.1223   59.5751   77.56
   5 │       4   8.81562      18.3516     8.65827   27.5859   48.6269   60.18 ⋯
   6 │       5   6.62888      10.4134    24.8194    26.0895   30.1605   52.30
   7 │       6  21.8397        6.63927   18.9708    32.892    43.1501   62.86
   8 │       7  53.343        26.798      6.63927   10.8879   17.2044   56.18
   9 │       8  25.6679       63.0572    22.1414    17.074    41.7448   60.27 ⋯
  10 │       9  12.827        69.2055    46.7629    13.9652   33.8874   61.82
  11 │      10   0.0          68.664     59.0435    33.3576   47.4528   57.83
  12 │      11   0.0          95.175     80.0382    59.8986   39.5848   50.28
  13 │      12   0.0         134.988    130.37      96.8632   75.7049   58.86 ⋯
  14 │      13   0.0          48.0892   142.556    140.378   154.726    87.93
  15 │      14   0.0          54.1941   146.384     99.4814   96.1941  102.94
  16 │      15   0.0           6.91966  195.171    132.525   135.234    89.85
  17 │      16   0.0          21.7387   123.134    176.741   158.27    137.23 ⋯
  18 │      17   0.0           0.0       62.0467   136.312   201.795   168.13
  19 │      18   0.0           0.0       79.6299   139.698   173.167   192.87
  20 │      19   0.0           0.0       33.2664    57.5364  167.224   196.48
  21 │      20   0.0           0.0       29.7794    54.9728  144.799   207.49 ⋯
  22 │      21   0.0           0.0       31.409    132.742   185.58    121.83
  23 │      22   0.0           0.0       19.5899    42.3036   96.2678  187.12
  24 │      23   0.00122968    3.00895    5.90986   33.5057  104.334   152.21
  25 │      24   0.0           5.62614    7.67626   63.1623   45.9976   79.56 ⋯
                                                             19 columns omitted

julia> z_data = Matrix{Float64}(df)'
25×25 adjoint(::Matrix{Float64}) with eltype Float64:
   0.0         1.0        2.0      …   22.0       23.0          24.0
  27.8099     27.7197    30.4267        0.0        0.00122968    0.0
  49.6194     48.5502    33.4775        0.0        3.00895       5.62614
  83.0807     65.2137    44.8095       19.5899     5.90986       7.67626
 116.663      95.2767    62.4749       42.3036    33.5057       63.1623
 130.414     116.996     77.4352   …   96.2678   104.334        45.9976
 150.721     133.906    104.215       187.121    152.216        79.5669
 220.187     152.341    102.739       179.663    198.199       227.311
 156.154     151.934    137.0         221.39     191.841       203.929
 148.642     160.114    186.071       154.262    228.735       172.562
 203.785     179.533    219.317    …  142.16     168.104       177.146
 206.039     147.618    181.762       148.574    144.276       140.455
 107.162     170.394    120.915        67.1794   110.744       123.99
  68.3697    121.819    143.184        40.6904    57.6521      110.346
  45.3359     52.5854    82.405        39.7451    42.635        65.1232
  49.9614     33.0887    48.4713   …   26.1017    27.9189       34.3189
  21.8928     38.4097    74.7146       14.4847    15.4105       24.5278
  17.0255     44.2484    60.0909        8.65873    8.0561        9.56107
  11.7432     69.5786     7.07353       3.89604    3.90283       3.33499
  14.7523      4.01935    6.08985       3.57139    3.87977       5.59049
  13.6671      3.05002    6.53745  …    3.89604    3.93672       5.48735
   5.67756     3.03972    6.6661        3.89604    3.96863       5.9095
   3.31234     2.99614    7.30696       3.89604    0.123626      5.86899
   1.15652     2.96795    5.73684       1.07776    3.98553       5.83382
  -0.147662    1.99959    3.62563       0.0       -0.183574      3.56818

julia> (sh_0, sh_1) = size(z_data)
(25, 25)

julia> x = range(0, stop=1, length=sh_0)
0.0:0.041666666666666664:1.0

julia> y = range(0, stop=1, length=sh_1)
0.0:0.041666666666666664:1.0

julia> layout = Layout(
           title="Mt Bruno Elevation", autosize=false,
           width=500, height=500,
           margin=attr(l=65, r=50, b=65, t=90)
       )
layout with fields autosize, height, margin, template, title, and width


julia> plot(surface(z=z_data, x=x, y=y), layout)




[ Info: Listening on: 127.0.0.1:9958, thread id: 1

julia> 

julia> 

julia> 

julia> 

julia> PlotlyJS.plot(surface(z=z_data), layout)

julia> PlotlyJS.surface(S,o, (S,o) -> L(S,o), color=:jet, xlabel="S", ylabel="o")
ERROR: MethodError: no method matching surface(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, ::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, ::var"#9#10"; color::Symbol, xlabel::String, ylabel::String)
Stacktrace:
 [1] top-level scope
   @ REPL[29]:1

julia> gr()
ERROR: UndefVarError: `gr` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[30]:1

julia> p2 = Plots.heatmap(S, 0, (S,0) -> L(S, 0), color=:jet, xlabel="S", ylabel="0",
       title="bird's eye view")
ERROR: syntax: "0" is not a valid function argument name around REPL[31]:1
Stacktrace:
 [1] top-level scope
   @ REPL[31]:1

julia> p2 = Plots.heatmap(S, Y, (S, Y) -> L(S, Y), color=:jet, xlabel="S", ylabel="Y", title="bird's eye view")
ERROR: UndefVarError: `Plots` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[32]:1

julia> SS = 25;

julia> vline!([SS],label=false,color=:black)
ERROR: UndefVarError: `vline!` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[34]:1

julia> p2 = Plots.heatmap(S, o,(S, o) -> L(S, o), color=:jet, xlabel="S", ylabel="0",title="bird's eye view")
ERROR: UndefVarError: `Plots` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[35]:1

julia> SS = 25;

julia> using Plots
WARNING: using Plots.surface in module Main conflicts with an existing identifier.
WARNING: using Plots.plot in module Main conflicts with an existing identifier.

julia> p2 = heatmap(S, o, (S, o) -> L(S, o), color=:jet, xlabel="S", ylabel="0", title="bird's eye view")
WARNING: both Plots and PlotlyJS export "heatmap"; uses of it in module Main must be qualified
ERROR: UndefVarError: `heatmap` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[38]:1

julia> using Plots

julia> p2 = heatmap(S, o, (S, o) -> L(S, o), color=:jet, xlabel="S", ylabel="0", title="bird's eye view")
ERROR: UndefVarError: `heatmap` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[40]:1

julia> using Plots

julia> p2 = Plots.heatmap(S, o,(S, o) -> L(S, o), color=:jet, xlabel="S", ylabel="0",title="bird's eye view")

julia> 

julia> SS = 25;

julia> vline!([SS],label=false,color=:black)

julia> p3=Plots.plot(o, o -> L(SS, o), label=false, xlabel='0', title="L(o|S=$SS)");

julia> Plots.plot(p2,p3)

julia> 

julia> savefig("Desktop/ML_S25.png")
WARNING: both Plots and PlotlyJS export "savefig"; uses of it in module Main must be qualified
ERROR: UndefVarError: `savefig` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[47]:1

julia> Plots.savefig("Desktop/ML_S25.png")
"/Users/shreyanshagrawal/Desktop/ML_S25.png"

julia> using Imageview : Imageview
ERROR: ArgumentError: Package Imageview not found in current path.
- Run `import Pkg; Pkg.add("Imageview")` to install the Imageview package.
Stacktrace:
 [1] macro expansion
   @ ./loading.jl:1595 [inlined]
 [2] macro expansion
   @ ./lock.jl:267 [inlined]
 [3] require(into::Module, mod::Symbol)
   @ Base ./loading.jl:1576
 [4] top-level scope
   @ REPL[49]:1

julia> g.add("Imageview")
ERROR: UndefVarError: `g` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[50]:1

julia> Pkg.add("Imageview")
ERROR: The following package names could not be resolved:
 * Imageview (not found in project, manifest or registry)
   Suggestions: ImageView TerminalImageViewer_jll DecisionMakingEnvironments
Stacktrace:
  [1] pkgerror(msg::String)
    @ Pkg.Types /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:69
  [2] ensure_resolved(ctx::Pkg.Types.Context, manifest::Pkg.Types.Manifest, pkgs::Vector{Pkg.Types.PackageSpec}; registry::Bool)
    @ Pkg.Types /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:995
  [3] ensure_resolved
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/Types.jl:946 [inlined]
  [4] add(ctx::Pkg.Types.Context, pkgs::Vector{Pkg.Types.PackageSpec}; preserve::Pkg.Types.PreserveLevel, platform::Base.BinaryPlatforms.Platform, kwargs::Base.Pairs{Symbol, Base.TTY, Tuple{Symbol}, NamedTuple{(:io,), Tuple{Base.TTY}}})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:264
  [5] add(pkgs::Vector{Pkg.Types.PackageSpec}; io::Base.TTY, kwargs::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:156
  [6] add(pkgs::Vector{Pkg.Types.PackageSpec})
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:145
  [7] #add#25
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:144 [inlined]
  [8] add
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:144 [inlined]
  [9] #add#24
    @ /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:143 [inlined]
 [10] add(pkg::String)
    @ Pkg.API /Applications/Julia-1.9.app/Contents/Resources/julia/share/julia/stdlib/v1.9/Pkg/src/API.jl:143
 [11] top-level scope
    @ REPL[51]:1

julia> add ImageView

ERROR: syntax: extra token "ImageView" after end of expression
Stacktrace:
 [1] top-level scope
   @ none:1

(@v1.9) pkg> add ImageView
   Resolving package versions...
   Installed Xorg_libXcomposite_jll ─ v0.4.5+4
   Installed hicolor_icon_theme_jll ─ v0.17.0+3
   Installed Pango_jll ────────────── v1.50.9+0
   Installed at_spi2_core_jll ─────── v2.34.0+4
   Installed OffsetArrays ─────────── v1.12.9
   Installed iso_codes_jll ────────── v4.11.0+0
   Installed AxisArrays ───────────── v0.4.6
   Installed Gtk ──────────────────── v1.3.0
   Installed GTK3_jll ─────────────── v3.24.31+0
   Installed ImageMetadata ────────── v0.9.8
   Installed adwaita_icon_theme_jll ─ v3.33.92+5
   Installed MappedArrays ─────────── v0.4.2
   Installed Graphics ─────────────── v1.1.2
   Installed Libepoxy_jll ─────────── v1.5.10+0
   Installed PaddedViews ──────────── v0.5.12
   Installed Xorg_libXtst_jll ─────── v1.2.3+4
   Installed RoundingIntegers ─────── v1.1.0
   Installed gdk_pixbuf_jll ───────── v2.42.8+0
   Installed at_spi2_atk_jll ──────── v2.34.1+4
   Installed Dbus_jll ─────────────── v1.12.16+3
   Installed MosaicViews ──────────── v0.3.4
   Installed ATK_jll ──────────────── v2.38.0+0
   Installed StackViews ───────────── v0.1.1
   Installed ImageCore ────────────── v0.9.4
   Installed AbstractFFTs ─────────── v1.3.2
   Installed Librsvg_jll ──────────── v2.54.5+0
   Installed RangeArrays ──────────── v0.3.2
   Installed SimpleTraits ─────────── v0.9.4
   Installed IntervalSets ─────────── v0.7.4
   Installed Adapt ────────────────── v3.6.2
   Installed ImageBase ────────────── v0.1.5
   Installed GtkObservables ───────── v1.2.9
   Installed IterTools ────────────── v1.8.0
   Installed Cairo ────────────────── v1.0.5
   Installed Xorg_libXdamage_jll ──── v1.1.5+4
   Installed ImageAxes ────────────── v0.6.10
   Installed MultiChannelColors ───── v0.1.2
   Installed ImageView ────────────── v0.11.6
  Downloaded artifact: iso_codes
  Downloaded artifact: hicolor_icon_theme
  Downloaded artifact: Pango
  Downloaded artifact: GTK3
  Downloaded artifact: adwaita_icon_theme
  Downloaded artifact: Libepoxy
  Downloaded artifact: gdk_pixbuf
  Downloaded artifact: ATK
  Downloaded artifact: Librsvg
    Updating `~/.julia/environments/v1.9/Project.toml`
  [86fae568] + ImageView v0.11.6
    Updating `~/.julia/environments/v1.9/Manifest.toml`
  [621f4979] + AbstractFFTs v1.3.2
  [79e6a3ab] + Adapt v3.6.2
  [39de3d68] + AxisArrays v0.4.6
  [159f3aea] + Cairo v1.0.5
  [a2bd30eb] + Graphics v1.1.2
  [4c0ca9eb] + Gtk v1.3.0
  [8710efd8] + GtkObservables v1.2.9
  [2803e5a7] + ImageAxes v0.6.10
  [c817782e] + ImageBase v0.1.5
  [a09fc81d] + ImageCore v0.9.4
  [bc367c6b] + ImageMetadata v0.9.8
  [86fae568] + ImageView v0.11.6
  [8197267c] + IntervalSets v0.7.4
  [c8e1da08] + IterTools v1.8.0
  [dbb5928d] + MappedArrays v0.4.2
  [e94cdb99] + MosaicViews v0.3.4
  [d4071afc] + MultiChannelColors v0.1.2
  [6fe1bfb0] + OffsetArrays v1.12.9
  [5432bcbf] + PaddedViews v0.5.12
  [b3c3ace0] + RangeArrays v0.3.2
  [d5f540fe] + RoundingIntegers v1.1.0
  [699a6c99] + SimpleTraits v0.9.4
  [cae243ae] + StackViews v0.1.1
  [7b86fcea] + ATK_jll v2.38.0+0
  [ee1fde0b] + Dbus_jll v1.12.16+3
  [77ec8976] + GTK3_jll v3.24.31+0
  [42c93a91] + Libepoxy_jll v1.5.10+0
  [925c91fb] + Librsvg_jll v2.54.5+0
  [36c8627f] + Pango_jll v1.50.9+0
  [3c9796d7] + Xorg_libXcomposite_jll v0.4.5+4
  [0aeada51] + Xorg_libXdamage_jll v1.1.5+4
  [b6f176f1] + Xorg_libXtst_jll v1.2.3+4
⌅ [b437f822] + adwaita_icon_theme_jll v3.33.92+5
  [de012916] + at_spi2_atk_jll v2.34.1+4
  [0fc3237b] + at_spi2_core_jll v2.34.0+4
  [da03df04] + gdk_pixbuf_jll v2.42.8+0
  [059c91fe] + hicolor_icon_theme_jll v0.17.0+3
  [bf975903] + iso_codes_jll v4.11.0+0
        Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`
Precompiling project...
  39 dependencies successfully precompiled in 42 seconds. 174 already precompiled.

julia> using ImageView:imshow

julia> lambda = Float64.(load("Desktop/cameraman.tif"))
ERROR: UndefVarError: `load` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[55]:1

julia> using Images
 │ Package Images not found, but a package named Images is available from a registry. 
 │ Install package?
 │   (@v1.9) pkg> add Images 
 └ (y/n/o) [y]: y
   Resolving package versions...
   Installed ImageIO ─────────────────── v0.6.6
   Installed ImageSegmentation ───────── v1.8.1
   Installed TiledIteration ──────────── v0.3.1
   Installed JpegTurbo ───────────────── v0.1.2
   Installed TiffImages ──────────────── v0.6.4
   Installed FFTW ────────────────────── v1.7.1
   Installed ImageMagick ─────────────── v1.2.1
   Installed PNGFiles ────────────────── v0.3.17
   Installed RegionTrees ─────────────── v0.3.2
   Installed Sixel ───────────────────── v0.1.2
   Installed ProgressMeter ───────────── v1.7.2
   Installed OpenEXR ─────────────────── v0.3.2
   Installed StaticArrays ────────────── v1.5.26
   Installed Images ──────────────────── v0.25.3
   Installed CEnum ───────────────────── v0.4.2
   Installed IntelOpenMP_jll ─────────── v2023.1.0+0
   Installed Netpbm ──────────────────── v1.1.0
   Installed MetaGraphs ──────────────── v0.7.2
   Installed Distances ───────────────── v0.10.8
   Installed NearestNeighbors ────────── v0.4.13
   Installed RealDot ─────────────────── v0.1.0
   Installed ArnoldiMethod ───────────── v0.2.0
   Installed JLD2 ────────────────────── v0.4.31
   Installed PkgVersion ──────────────── v0.3.2
   Installed QOI ─────────────────────── v1.0.0
   Installed FileIO ──────────────────── v1.16.1
   Installed StaticArraysCore ────────── v1.4.0
   Installed CustomUnitRanges ────────── v1.0.2
   Installed Ratios ──────────────────── v0.4.5
   Installed ComputationalResources ──── v0.3.2
   Installed CatIndices ──────────────── v0.2.2
   Installed ImageMagick_jll ─────────── v6.9.12+4
   Installed LazyModules ─────────────── v0.3.1
   Installed Graphs ──────────────────── v1.8.0
   Installed ImageTransformations ────── v0.9.5
   Installed Inflate ─────────────────── v0.1.3
   Installed ImageShow ───────────────── v0.3.7
   Installed ImageQualityIndexes ─────── v0.3.6
   Installed Imath_jll ───────────────── v3.1.7+0
   Installed MKL_jll ─────────────────── v2023.1.0+0
   Installed IntegralArrays ──────────── v0.1.5
   Installed CoordinateTransformations ─ v0.6.3
   Installed ChainRulesCore ──────────── v1.16.0
   Installed Ghostscript_jll ─────────── v9.55.0+4
   Installed Quaternions ─────────────── v0.7.4
   Installed LittleCMS_jll ───────────── v2.12.0+0
   Installed FFTW_jll ────────────────── v3.3.10+0
   Installed Rotations ───────────────── v1.5.1
   Installed OpenEXR_jll ─────────────── v3.1.4+0
   Installed OpenJpeg_jll ────────────── v2.4.0+0
   Installed IndirectArrays ──────────── v1.0.0
   Installed libsixel_jll ────────────── v1.10.3+0
   Installed ImageDistances ──────────── v0.2.16
   Installed SimpleWeightedGraphs ────── v1.4.0
   Installed AxisAlgorithms ──────────── v1.0.1
   Installed WoodburyMatrices ────────── v0.5.5
   Installed ImageMorphology ─────────── v0.3.2
   Installed FFTViews ────────────────── v0.3.2
   Installed Clustering ──────────────── v0.15.3
   Installed ImageContrastAdjustment ─── v0.3.10
   Installed Interpolations ──────────── v0.14.7
   Installed ImageFiltering ──────────── v0.7.5
  Downloaded artifact: ImageMagick
  Downloaded artifact: Imath
  Downloaded artifact: Ghostscript
  Downloaded artifact: LittleCMS
  Downloaded artifact: FFTW
  Downloaded artifact: OpenEXR
  Downloaded artifact: OpenJpeg
  Downloaded artifact: libsixel
    Updating `~/.julia/environments/v1.9/Project.toml`
  [916415d5] + Images v0.25.3
    Updating `~/.julia/environments/v1.9/Manifest.toml`
  [ec485272] + ArnoldiMethod v0.2.0
  [13072b0f] + AxisAlgorithms v1.0.1
  [fa961155] + CEnum v0.4.2
  [aafaddc9] + CatIndices v0.2.2
  [d360d2e6] + ChainRulesCore v1.16.0
  [aaaa29a8] + Clustering v0.15.3
  [ed09eef8] + ComputationalResources v0.3.2
  [150eb455] + CoordinateTransformations v0.6.3
  [dc8bdbbb] + CustomUnitRanges v1.0.2
  [b4f34e82] + Distances v0.10.8
  [4f61f5a4] + FFTViews v0.3.2
  [7a1cc6ca] + FFTW v1.7.1
  [5789e2e9] + FileIO v1.16.1
  [86223c79] + Graphs v1.8.0
  [f332f351] + ImageContrastAdjustment v0.3.10
  [51556ac3] + ImageDistances v0.2.16
  [6a3955dd] + ImageFiltering v0.7.5
  [82e4d734] + ImageIO v0.6.6
⌃ [6218d12a] + ImageMagick v1.2.1
⌅ [787d08f9] + ImageMorphology v0.3.2
  [2996bd0c] + ImageQualityIndexes v0.3.6
  [80713f31] + ImageSegmentation v1.8.1
  [4e3cecfd] + ImageShow v0.3.7
  [02fcd773] + ImageTransformations v0.9.5
  [916415d5] + Images v0.25.3
  [9b13fd28] + IndirectArrays v1.0.0
  [d25df0c9] + Inflate v0.1.3
  [1d092043] + IntegralArrays v0.1.5
  [a98d9a8b] + Interpolations v0.14.7
  [033835bb] + JLD2 v0.4.31
  [b835a17e] + JpegTurbo v0.1.2
  [8cdb02fc] + LazyModules v0.3.1
  [626554b9] + MetaGraphs v0.7.2
  [b8a86587] + NearestNeighbors v0.4.13
  [f09324ee] + Netpbm v1.1.0
  [52e1d378] + OpenEXR v0.3.2
  [f57f5aa1] + PNGFiles v0.3.17
  [eebad327] + PkgVersion v0.3.2
  [92933f4c] + ProgressMeter v1.7.2
  [4b34888f] + QOI v1.0.0
  [94ee1d12] + Quaternions v0.7.4
  [c84ed2f1] + Ratios v0.4.5
  [c1ae055f] + RealDot v0.1.0
  [dee08c22] + RegionTrees v0.3.2
  [6038ab10] + Rotations v1.5.1
  [47aef6b3] + SimpleWeightedGraphs v1.4.0
  [45858cf5] + Sixel v0.1.2
  [90137ffa] + StaticArrays v1.5.26
  [1e83bf80] + StaticArraysCore v1.4.0
  [731e570b] + TiffImages v0.6.4
⌅ [06e1c1a7] + TiledIteration v0.3.1
  [efce3f68] + WoodburyMatrices v0.5.5
  [f5851436] + FFTW_jll v3.3.10+0
  [61579ee1] + Ghostscript_jll v9.55.0+4
⌅ [c73af94c] + ImageMagick_jll v6.9.12+4
  [905a6f67] + Imath_jll v3.1.7+0
  [1d5cc7b8] + IntelOpenMP_jll v2023.1.0+0
  [d3a379c0] + LittleCMS_jll v2.12.0+0
  [856f044c] + MKL_jll v2023.1.0+0
  [18a262bb] + OpenEXR_jll v3.1.4+0
  [643b3616] + OpenJpeg_jll v2.4.0+0
  [075b6546] + libsixel_jll v1.10.3+0
  [4af54fe1] + LazyArtifacts
  [1a1011a3] + SharedArrays
        Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Precompiling project...
  69 dependencies successfully precompiled in 67 seconds. 213 already precompiled.

julia> imshow(lambdahat)
ERROR: UndefVarError: `lambdahat` not defined
Stacktrace:
 [1] top-level scope
   @ REPL[57]:1

julia> lambda = Float64.(load("Desktop/cameraman.tif")))
ERROR: syntax: extra token ")" after end of expression
Stacktrace:
 [1] top-level scope
   @ none:1

julia> lambda = Float64.(load("Desktop/cameraman.tif"))
ERROR: ArgumentError: No file exists at given path: Desktop/cameraman.tif
Stacktrace:
 [1] checkpath_load
   @ ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:167 [inlined]
 [2] load(::String; options::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
   @ FileIO ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:110
 [3] load(::String)
   @ FileIO ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:109
 [4] top-level scope
   @ REPL[59]:1

julia> lambda = Float64.(load("Desktop/cameraman.tif"))
ERROR: ArgumentError: No file exists at given path: Desktop/cameraman.tif
Stacktrace:
 [1] checkpath_load
   @ ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:167 [inlined]
 [2] load(::String; options::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
   @ FileIO ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:110
 [3] load(::String)
   @ FileIO ~/.julia/packages/FileIO/BE7iZ/src/loadsave.jl:109
 [4] top-level scope
   @ REPL[59]:1

julia> using Plots;

julia> gr();

julia> p2=Plots.heatmap(S,o,(S,o)->L(S,o),color=:jet, xlabel="S", ylabel="o",title="Bird eye view");

julia> SS=25;

julia> vline!([SS],label=false,color=:black);

julia> P3=Plots.plot(o,o->L(SS,o),label=false, xlabel='o',title="L(o|S=$SS)");

julia> Plots.plot(p2,P3)

julia> savefig("julia_fig.png")

