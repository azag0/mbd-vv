set spellfile=.utf8-8.add
set spelllang=en_us
let &makeprg = "make 2> /dev/null | texoutparse.py -nl"
set errorformat=
            \%-G/usr/local/texlive/%m,
            \%-GROOT:%m,
            \%-GLatexmk:%m,
            \%-G%f:LaTeX2e\ <%m,
            \%-G%f:Babel\ <%m,
            \%-G%f:%\\s%#ABD:\ %m,
            \%-G%f:*geometry*\ %m,
            \%-G%f:%.%#achemso-control%m,
            \%-G%f:Underfull%m,
            \%-G%f:Overfull%m,
            \%-G%f:Package\ hyperref\ Message:%m,
            \%-G%f:Package\ hyperref\ Warning:%m,
            \%-G%f:Package\ rerunfilecheck\ Warning:%m,
            \%-G%f:Package\ natbib\ Warning:\ There\ were\ undefined\ citations%m,
            \%-G%f:%.%#multiple\ pdfs\ with\ page\ group%m,
            \%-G%f:LaTeX\ Warning:\ A\ float\ is\ stuck%m,
            \%f:%\\%%(Package\ %\\w%#\ Warning%\\)%\\@=%m\ on\ input\ line\ %l.,
            \%f:%\\%%(Package\ %\\w%#\ Warning%\\)%\\@=%m,
            \%f:%\\%%(Package\ %\\w%#\ Message%\\)%\\@=%m,
            \%f:%\\%%(LaTeX\ %\\w%#\ Message%\\)%\\@=%m,
            \%f:%\\%%(LaTeX\ Warning%\\)%\\@=%m\ on\ input\ line\ %l.,
            \RuntimeError:\ %m,
            \%E%>%f:!\ %m,
            \%C%f:l.%l\ %m,
            \%C%f:%m
