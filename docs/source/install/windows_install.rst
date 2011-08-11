.. _windows_install:

---------------------------
Windows Installation
---------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Supported platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PySPH is tested on the following operating system versions:

 + Windows 7 (32 and 64 bit)
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
The following packages need to be installed on your system. Contact
your system administrator if you do not have administrative rights.

+ Enthought Python Distribution (EPD_)
+ TortoiseHG from http://tortoisehg.bitbucket.org

.. _EPD: http://code.enthought.com

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Download the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure TortoiseHG to clone the PySPH repository:

 + Right click anywhere in the desktop.
 + Scroll down to TortoiseHG->clone
 + Enter as :guilabel:`Source` :token:`pysph.googlecode.com/hg`
 + Enter as :guilabel:`Destination` a directory of your choice.
 + Click on :guilabel:`clone`
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing PySPH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming all steps hitherto completed without errors, you can install PySPH like so:

 + Use the Windows shell to navigate to your PySPH clone.
 + Run :command:`python setup.py install`
 
After the installation, you should test the installation by running the 
tests (see :doc:`tests`)