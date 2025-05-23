$env:PIGLIT_NO_FAST_SKIP = 1
$env:PIGLIT_SPIRV_AS_BINARY = "C:\VulkanSDK\$env:VULKAN_SDK_VERSION\Bin\spirv-as.exe"

Copy-Item -Path _install\bin\opengl32.dll -Destination C:\Piglit\bin\opengl32.dll
Copy-Item -Path _install\bin\libgallium_wgl.dll -Destination C:\Piglit\bin\libgallium_wgl.dll

$jobs = ""
if ($null -ne $env:FDO_CI_CONCURRENT) {
  $jobs = "--jobs", "$($env:FDO_CI_CONCURRENT)"
}

deqp-runner suite --output .\logs --suite "_install/$env:PIGLIT_SUITE" `
  --skips "_install/$env:PIGLIT_SKIPS" `
  --baseline "_install/$env:PIGLIT_BASELINE" `
  --flakes "_install/$env:PIGLIT_FLAKES" `
  $jobs
if (!$?) {
  Exit 1
}
