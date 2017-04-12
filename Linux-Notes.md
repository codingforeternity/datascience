#### Remote desktop (xrdp; 4/12/17)
* [Arch Linux xrdp wiki page](https://wiki.archlinux.org/index.php/xrdp)
  * except it says you need to start xrdp-sesman service, which now gets automatically started w/ xrdp
* [These](http://c-nergy.be/blog/?p=5305) are the instructions I followed to install xrdp and perform initial configuration.
  * But they don't explain how to connect to an existing session.
* [These](http://c-nergy.be/blog/?p=4471) instructions might explain how to connect to an existing session.
  * Or [these](http://askubuntu.com/questions/235905/use-xrdp-to-connect-to-desktop-session) instructions might be more straightforward.  They connect to Vino (VNC server) on the Ubuntu machine via an xrdp-to-vino gateway.

#### .bashrc vs. .bash_profile
* .bashrc is only sourced for non-login shells, like terminals that you start once you've already logged in
* .bash_profile is sourced when you login
* "Most of the time you don’t want to maintain two separate config files for login and non-login shells--when you set a PATH, you want it to apply to both. You can fix this by sourcing .bashrc from your .bash_profile file, then putting PATH and common settings in .bashrc.  To do this, add the following lines to .bash_profile:
  * `if [ -f ~/.bashrc ]; then`
  * `source ~/.bashrc`
  * `fi`
  * [http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html]
* wrt IntelliJ specifically: http://askubuntu.com/questions/542152/desktop-file-with-bashrc-environment

#### [Ubuntu Sound](https://wiki.ubuntu.com/Sound)
  * The issue with my sound after upgrading to Ubuntu 16.04 was that the sound control is only shown for the applications that are currently running.  So to control the sound for Firefox, you have to first start Firefox.
  * Also see this command: `pavucontrol`

#### [UNIX Shell Tutorial](http://swcarpentry.github.io/shell-novice/)
* "put together by the always excellent Software Carpentry Foundation." [http://localhost:8888/notebooks/01.05-IPython-And-Shell-Commands.ipynb]

#### [Useful Linux commands](http://www.commandlinefu.com/commands/browse/sort-by-votes) (5/16/16)
* tips and tricks