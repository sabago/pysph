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
        
It is recommended to use the Enthought Python Distribution (EPD_) to
run Pysph on Windows. Contact your system administrator if you do not
have administrative rights.

Installing PySPH is now as easy as::

	   easy_install pysph
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing PySPH from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To check out a development version of the repository, you will need to
install TortoiseHG_

Configure TortoiseHG to clone the PySPH repository:

 + Right click anywhere in the desktop.
 + Scroll down to TortoiseHG->clone
 + Enter as :guilabel:`Source` :token:`pysph.googlecode.com/hg`
 + Enter as :guilabel:`Destination` a directory of your choice.
 + Click on :guilabel:`clone`

Now, you can install PySPH like so:

 + Use the Windows shell to navigate to your PySPH clone.
 + Run :command:`python setup.py install`
 
After the installation, you should test the installation by running the 
tests (see :doc:`tests`)


..  _TortoiseHG: http://tortoisehg.bitbucket.org

.. _EPD: http://code.enthought.com
