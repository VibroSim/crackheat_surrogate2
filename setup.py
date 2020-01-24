import sys
#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import subprocess
import re
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
import setuptools.command.bdist_egg
import sys
import distutils.spawn
#from Cython.Build import cythonize




class install_lib_save_version(install_lib):
    """Save version information"""
    def run(self):
        install_lib.run(self)
        
        for package in self.distribution.command_obj["build_py"].packages:
            install_dir=os.path.join(*([self.install_dir] + package.split('.')))
            fh=open(os.path.join(install_dir,"version.txt"),"w")
            fh.write("%s\n" % (version))  # version global, as created below
            fh.close()
            pass
        pass
    pass



# Extract GIT version
if os.path.exists(".git") and distutils.spawn.find_executable("git") is not None:
    # Check if tree has been modified
    modified = subprocess.call(["git","diff-index","--quiet","HEAD","--"]) != 0
    
    gitrev = subprocess.check_output(["git","rev-parse","HEAD"]).strip()

    version = "git-%s" % (gitrev)

    # See if we can get a more meaningful description from "git describe"
    try:
        versionraw=subprocess.check_output(["git","describe","--tags","--match=v*"],stderr=subprocess.STDOUT).decode('utf-8').strip()
        # versionraw is like v0.1.0-50-g434343
        # for compatibility with PEP 440, change it to
        # something like 0.1.0+50.g434343
        matchobj=re.match(r"""v([^.]+[.][^.]+[.][^-.]+)(-.*)?""",versionraw)
        version=matchobj.group(1)
        if matchobj.group(2) is not None:
            version += '+'+matchobj.group(2)[1:].replace("-",".")
            pass
        pass
    except subprocess.CalledProcessError:
        # Ignore error, falling back to above version string
        pass

    if modified and version.find('+') >= 0:
        version += ".modified"
        pass
    elif modified:
        version += "+modified"
        pass
    pass
else:
    version = "UNKNOWN"
    pass

print("version = %s" % (version))

crackheat_surrogate2_package_files = [ "pt_steps/*","Rscripts/*" ]

#ext_modules=cythonize("crackheat_surrogate2/*.pyx")
ext_modules=[]
em_dict=dict([ (module.name,module) for module in ext_modules])
#sca_pyx_ext=em_dict["crackclosuresim2.soft_closure_accel"]
#sca_pyx_ext.include_dirs=["."]
##sca_pyx_ext.extra_compile_args=['-O0','-g','-Wno-uninitialized']
#sca_pyx_ext.extra_compile_args=['-fopenmp','-O5','-Wno-uninitialized']
#sca_pyx_ext.libraries=['gomp']



console_scripts=[ "train_crackheat_surrogate2" ]
console_scripts_entrypoints = [ "%s = crackheat_surrogate2.bin.%s:main" % (script,script.replace("-","_")) for script in console_scripts ]



setup(name="crackheat_surrogate2",
      description="Surrogate model fitting for vibrothermography crack heating model, version 2",
      author="Stephen D. Holland",
      version=version,
      url="http://thermal.cnde.iastate.edu",
      zip_safe=False,
      ext_modules=ext_modules,
      packages=["crackheat_surrogate2","crackheat_surrogate2.bin"],
      cmdclass={"install_lib": install_lib_save_version },
      package_data={"crackheat_surrogate2": crackheat_surrogate2_package_files},
      entry_points={ "limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = crackheat_surrogate2:getstepurlpath" ],
                     "console_scripts": console_scripts_entrypoints,
                 })

