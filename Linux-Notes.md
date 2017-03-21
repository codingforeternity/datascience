#### .bashrc vs. .bash_profile
* .bashrc is only sourced for non-login shells, like terminals that you start once you've already logged in
* .bash_profile is sourced when you login
* "Most of the time you don’t want to maintain two separate config files for login and non-login shells--when you set a PATH, you want it to apply to both. You can fix this by sourcing .bashrc from your .bash_profile file, then putting PATH and common settings in .bashrc.  To do this, add the following lines to .bash_profile:
  * ```if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi```
  * [http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html]
* wrt IntelliJ specifically: http://askubuntu.com/questions/542152/desktop-file-with-bashrc-environment

#### [Ubuntu Sound](https://wiki.ubuntu.com/Sound)

#### [UNIX Shell Tutorial](http://swcarpentry.github.io/shell-novice/)
* "put together by the always excellent Software Carpentry Foundation." [http://localhost:8888/notebooks/01.05-IPython-And-Shell-Commands.ipynb]

#### [Useful Linux commands](http://www.commandlinefu.com/commands/browse/sort-by-votes) (5/16/16)
* tips and tricks