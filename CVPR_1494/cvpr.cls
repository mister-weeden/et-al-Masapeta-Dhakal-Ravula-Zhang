% ---------------------------------------------------------------
%
% The last version of CVPR latex template has been developed by
% Paolo.Ienne@di.epfl.ch and awf@acm.org about 15 years ago. 
% The previous version suffer from several issues:
% 1. Authors needs several individual files: cvpr.sty, cvpr_eso.sty
%    eso-pic.sty.
% 2. For cvpr rebuttal another verison of cvpr.sty is required.
% 3. Several warnings exists due to depreiciated options.
%
% This style file is intended to be used as a single style file 
% that could help to fuild review, rebuttal and final files with
% a single style file.
% If you find any bug or have any further suggestions, please contact
% Ming-Ming Cheng (cmm@nankai.edu.cn).
% 
% Acknowledge: this file is modified from the the old cvpr template 
% files: cvpr.sty, cvpr_eso.sty, and eso-pic.sty.
%
% https://www.overleaf.com/learn/latex/Writing_your_own_class
% ---------------------------------------------------------------




% ---------------------------------------------------------------
%
% $Id: cvpr.sty,v 1.3 2005/10/24 19:56:15 awf Exp $
% by Paolo.Ienne@di.epfl.ch some mods by awf@acm.org
%
% ---------------------------------------------------------------
%It 
% no guarantee is given that the format corresponds perfectly to
% IEEE 8.5" x 11" Proceedings, but most features should be ok.
%
% ---------------------------------------------------------------
% with LaTeX2e:
% =============
%
% use as
%   \documentclass[times,10pt,twocolumn]{article}
%   \usepackage{latex8}
%   \usepackage{times}
%
% ---------------------------------------------------------------

% with LaTeX 2.09:
% ================
%
% use as
%   \documentstyle[times,art10,twocolumn,latex8]{article}
%
% ---------------------------------------------------------------
% with both versions:
% ===================
%
% specify \cvprfinalcopy to emit the final camera-ready copy
%
% specify references as
%   \bibliographystyle{ieee}
%   \bibliography{...your files...}
%
% ---------------------------------------------------------------



% ---------------------------------------------------------------
%
%\usepackage{eso-pic}
%
%%
%% This is file `eso-pic.sty',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% eso-pic.dtx  (with options: `package')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 1998-2002 by Rolf Niepraschk <niepraschk@ptb.de>
%% 
%% This file may be distributed and/or modified under the conditions of
%% the LaTeX Project Public License, either version 1.2 of this license
%% or (at your option) any later version.  The latest version of this
%% license is in:
%% 
%%    http://www.latex-project.org/lppl.txt
%% 
%% and version 1.2 or later is part of all distributions of LaTeX version
%% 1999/12/01 or later.
%% 


% 
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesPackage{cvpr} [2020/5/16 Example LaTex class for IEEE CVPR]
\LoadClass[10pt,twocolumn,letterpaper]{article}

\RequirePackage{xspace}
\RequirePackage{silence}
\hbadness=10000 \vbadness=10000 \vfuzz=30pt \hfuzz=30pt
\RequirePackage{etoolbox}

\newtoggle{cvprrebuttal}
\newtoggle{cvprfinal}
\togglefalse{cvprfinal}
\DeclareOption{review}{\togglefalse{cvprrebuttal}}
\DeclareOption{rebuttal}{\toggletrue{cvprrebuttal}}
\DeclareOption{final}{\toggletrue{cvprfinal}}


\ProcessOptions\relax


% ---------------------------------------------------------------
%\input{everyshi.sty}
\newcommand{\@EveryShipout@Hook}{}
\newcommand{\@EveryShipout@AtNextHook}{}
\newcommand*{\EveryShipout}[1]
   {\g@addto@macro\@EveryShipout@Hook{#1}}
\newcommand*{\AtNextShipout}[1]
   {\g@addto@macro\@EveryShipout@AtNextHook{#1}}
\newcommand{\@EveryShipout@Shipout}{%
   \afterassignment\@EveryShipout@Test
   \global\setbox\@cclv= %
   }
\newcommand{\@EveryShipout@Test}{%
   \ifvoid\@cclv\relax
      \aftergroup\@EveryShipout@Output
   \else
      \@EveryShipout@Output
   \fi%
   }
\newcommand{\@EveryShipout@Output}{%
   \@EveryShipout@Hook%
   \@EveryShipout@AtNextHook%
      \gdef\@EveryShipout@AtNextHook{}%
   \@EveryShipout@Org@Shipout\box\@cclv%
   }
\newcommand{\@EveryShipout@Org@Shipout}{}
\newcommand*{\@EveryShipout@Init}{%
   \message{ABD: EveryShipout initializing macros}%
   \let\@EveryShipout@Org@Shipout\shipout
   \let\shipout\@EveryShipout@Shipout
   }
\AtBeginDocument{\@EveryShipout@Init}

% ---------------------------------------------------------------



\newcommand\LenToUnit[1]{#1\@gobble}

\newcommand\AtPageUpperLeft[1]{%
  \begingroup
    \@tempdima=0pt\relax\@tempdimb=\ESO@yoffsetI\relax
    \put(\LenToUnit{\@tempdima},\LenToUnit{\@tempdimb}){#1}%
  \endgroup
}
\newcommand\AtPageLowerLeft[1]{\AtPageUpperLeft{%
  \put(0,\LenToUnit{-\paperheight}){#1}}}
\newcommand\AtPageCenter[1]{\AtPageUpperLeft{%
  \put(\LenToUnit{.5\paperwidth},\LenToUnit{-.5\paperheight}){#1}}%
}
\newcommand\AtTextUpperLeft[1]{%
  \begingroup
    \setlength\@tempdima{1in}%
    \ifodd\c@page%
      \advance\@tempdima\oddsidemargin%
    \else%
      \advance\@tempdima\evensidemargin%
    \fi%
    \@tempdimb=\ESO@yoffsetI\relax\advance\@tempdimb-1in\relax%
    \advance\@tempdimb-\topmargin%
    \advance\@tempdimb-\headheight\advance\@tempdimb-\headsep%
    \put(\LenToUnit{\@tempdima},\LenToUnit{\@tempdimb}){#1}%
  \endgroup
}
\newcommand\AtTextLowerLeft[1]{\AtTextUpperLeft{%
  \put(0,\LenToUnit{-\textheight}){#1}}}
\newcommand\AtTextCenter[1]{\AtTextUpperLeft{%
  \put(\LenToUnit{.5\textwidth},\LenToUnit{-.5\textheight}){#1}}}
\newcommand{\ESO@HookI}{} \newcommand{\ESO@HookII}{}
\newcommand{\ESO@HookIII}{}
\newcommand{\AddToShipoutPicture}{%
  \@ifstar{\g@addto@macro\ESO@HookII}{\g@addto@macro\ESO@HookI}}
\newcommand{\ClearShipoutPicture}{\global\let\ESO@HookI\@empty}
\newcommand\ESO@isMEMOIR[1]{}
\@ifclassloaded{memoir}{\renewcommand\ESO@isMEMOIR[1]{#1}}{}
\newcommand{\@ShipoutPicture}{%
  \bgroup
    \@tempswafalse%
    \ifx\ESO@HookI\@empty\else\@tempswatrue\fi%
    \ifx\ESO@HookII\@empty\else\@tempswatrue\fi%
    \ifx\ESO@HookIII\@empty\else\@tempswatrue\fi%
    \if@tempswa%
      \@tempdima=1in\@tempdimb=-\@tempdima%
      \advance\@tempdimb\ESO@yoffsetI%
      \ESO@isMEMOIR{%
        \advance\@tempdima\trimedge%
        \advance\@tempdima\paperwidth%
        \advance\@tempdima-\stockwidth%
        \if@twoside\ifodd\c@page\else%
          \advance\@tempdima-2\trimedge%
          \advance\@tempdima-\paperwidth%
          \advance\@tempdima\stockwidth%
        \fi\fi%
        \advance\@tempdimb\trimtop}%
      \unitlength=1pt%
      \global\setbox\@cclv\vbox{%
        \vbox{\let\protect\relax
          \pictur@(0,0)(\strip@pt\@tempdima,\strip@pt\@tempdimb)%
            \ESO@HookIII\ESO@HookI\ESO@HookII%
            \global\let\ESO@HookII\@empty%
          \endpicture}%
          \nointerlineskip%
        \box\@cclv}%
    \fi
  \egroup
}
\EveryShipout{\@ShipoutPicture}
\RequirePackage{keyval}
\newif\ifESO@dvips\ESO@dvipsfalse \newif\ifESO@grid\ESO@gridfalse
\newif\ifESO@texcoord\ESO@texcoordfalse
\newcommand*\ESO@gridunitname{}
\newcommand*\ESO@gridunit{}
\newcommand*\ESO@labelfactor{}
\newcommand*\ESO@griddelta{}\newcommand*\ESO@griddeltaY{}
\newcommand*\ESO@gridDelta{}\newcommand*\ESO@gridDeltaY{}
\newcommand*\ESO@gridcolor{}
\newcommand*\ESO@subgridcolor{}
\newcommand*\ESO@subgridstyle{dotted}% ???
\newcommand*\ESO@gap{}
\newcommand*\ESO@yoffsetI{}\newcommand*\ESO@yoffsetII{}
\newcommand*\ESO@gridlines{\thinlines}
\newcommand*\ESO@subgridlines{\thinlines}
\newcommand*\ESO@hline[1]{\ESO@subgridlines\line(1,0){#1}}
\newcommand*\ESO@vline[1]{\ESO@subgridlines\line(0,1){#1}}
\newcommand*\ESO@Hline[1]{\ESO@gridlines\line(1,0){#1}}
\newcommand*\ESO@Vline[1]{\ESO@gridlines\line(0,1){#1}}
\newcommand\ESO@fcolorbox[4][]{\fbox{#4}}
\newcommand\ESO@color[1]{}
\newcommand\ESO@colorbox[3][]{%
  \begingroup
    \fboxrule=0pt\fbox{#3}%
  \endgroup
}
\newcommand\gridSetup[6][]{%
  \edef\ESO@gridunitname{#1}\edef\ESO@gridunit{#2}
  \edef\ESO@labelfactor{#3}\edef\ESO@griddelta{#4}
  \edef\ESO@gridDelta{#5}\edef\ESO@gap{#6}}
\define@key{ESO}{texcoord}[true]{\csname ESO@texcoord#1\endcsname}
\define@key{ESO}{pscoord}[true]{\csname @tempswa#1\endcsname
  \if@tempswa\ESO@texcoordfalse\else\ESO@texcoordtrue\fi}
\define@key{ESO}{dvips}[true]{\csname ESO@dvips#1\endcsname}
\define@key{ESO}{grid}[true]{\csname ESO@grid#1\endcsname
  \setkeys{ESO}{gridcolor=black,subgridcolor=black}}
\define@key{ESO}{colorgrid}[true]{\csname ESO@grid#1\endcsname
  \setkeys{ESO}{gridcolor=red,subgridcolor=green}}
\define@key{ESO}{gridcolor}{\def\ESO@gridcolor{#1}}
\define@key{ESO}{subgridcolor}{\def\ESO@subgridcolor{#1}}
\define@key{ESO}{subgridstyle}{\def\ESO@subgridstyle{#1}}%
\define@key{ESO}{gridunit}{%
  \def\@tempa{#1}
  \def\@tempb{bp}
  \ifx\@tempa\@tempb
    \gridSetup[\@tempa]{1bp}{1}{10}{50}{2}
  \else
    \def\@tempb{pt}
    \ifx\@tempa\@tempb
      \gridSetup[\@tempa]{1pt}{1}{10}{50}{2}
    \else
      \def\@tempb{in}
      \ifx\@tempa\@tempb
        \gridSetup[\@tempa]{.1in}{.1}{2}{10}{.5}
      \else
        \gridSetup[mm]{1mm}{1}{5}{20}{1}
      \fi
    \fi
  \fi
}
%\setkeys{ESO}{subgridstyle=solid,pscoord=true,gridunit=mm}
% \def\ProcessOptionsWithKV#1{%
%   \let\@tempc\@empty
%   \@for\CurrentOption:=\@classoptionslist\do{%
%     \@ifundefined{KV@#1@\CurrentOption}%
%     {}{\edef\@tempc{\@tempc,\CurrentOption,}}}%
%   \edef\@tempc{%
%     \noexpand\setkeys{#1}{\@tempc\@ptionlist{\@currname.\@currext}}
%   }%
%   \@tempc
%   \AtEndOfPackage{\let\@unprocessedoptions\relax}}%
%\ProcessOptionsWithKV{ESO}%
\newcommand\ESO@div[2]{%
  \@tempdima=#1\relax\@tempdimb=\ESO@gridunit\relax
  \@tempdimb=#2\@tempdimb\divide\@tempdima by \@tempdimb%
  \@tempcnta\@tempdima\advance\@tempcnta\@ne}
\AtBeginDocument{%
  \IfFileExists{color.sty}
  {%
    \RequirePackage{color}
    \let\ESO@color=\color\let\ESO@colorbox=\colorbox
    \let\ESO@fcolorbox=\fcolorbox
  }{}
  \@ifundefined{Gin@driver}{}%
  {%
    \ifx\Gin@driver\@empty\else%
      \filename@parse{\Gin@driver}\def\reserved@a{dvips}%
      \ifx\filename@base\reserved@a\ESO@dvipstrue\fi%
    \fi
  }%
  \ifx\pdfoutput\undefined\else
    \ifx\pdfoutput\relax\else
      \ifcase\pdfoutput\else
        \ESO@dvipsfalse%
      \fi
    \fi
  \fi
  \ifESO@dvips\def\@tempb{eepic}\else\def\@tempb{epic}\fi
  \def\@tempa{dotted}%\def\ESO@gap{\LenToUnit{6\@wholewidth}}%
  \ifx\@tempa\ESO@subgridstyle
    \IfFileExists{\@tempb.sty}%
    {%
      \RequirePackage{\@tempb}
      \renewcommand*\ESO@hline[1]{\ESO@subgridlines\dottedline{\ESO@gap}%
        (0,0)(##1,0)}
      \renewcommand*\ESO@vline[1]{\ESO@subgridlines\dottedline{\ESO@gap}%
        (0,0)(0,##1)}
    }{}
  \else
    \ifx\ESO@gridcolor\ESO@subgridcolor%
      \renewcommand*\ESO@gridlines{\thicklines}
    \fi
  \fi
}
\ifESO@texcoord
  \def\ESO@yoffsetI{0pt}\def\ESO@yoffsetII{-\paperheight}
  \edef\ESO@griddeltaY{-\ESO@griddelta}\edef\ESO@gridDeltaY{-\ESO@gridDelta}
\else
  \def\ESO@yoffsetI{\paperheight}\def\ESO@yoffsetII{0pt}
  \edef\ESO@griddeltaY{\ESO@griddelta}\edef\ESO@gridDeltaY{\ESO@gridDelta}
\fi
\newcommand\ESO@gridpicture{%
  \begingroup
    \setlength\unitlength{\ESO@gridunit}%
    \ESO@color{\ESO@subgridcolor}%
    \ESO@div{\paperheight}{\ESO@griddelta}%
    \multiput(0,0)(0,\ESO@griddeltaY){\@tempcnta}%
      {\ESO@hline{\LenToUnit{\paperwidth}}}%
    \ESO@div{\paperwidth}{\ESO@griddelta}%
    \multiput(0,\LenToUnit{\ESO@yoffsetII})(\ESO@griddelta,0){\@tempcnta}%
      {\ESO@vline{\LenToUnit{\paperheight}}}%
    \ESO@color{\ESO@gridcolor}%
    \ESO@div{\paperheight}{\ESO@gridDelta}%
    \multiput(0,0)(0,\ESO@gridDeltaY){\@tempcnta}%
      {\ESO@Hline{\LenToUnit{\paperwidth}}}%
    \ESO@div{\paperwidth}{\ESO@gridDelta}%
    \multiput(0,\LenToUnit{\ESO@yoffsetII})(\ESO@gridDelta,0){\@tempcnta}%
      {\ESO@Vline{\LenToUnit{\paperheight}}}%
    \fontsize{10}{12}\normalfont%
    \ESO@div{\paperwidth}{\ESO@gridDelta}%
    \multiput(0,\ESO@gridDeltaY)(\ESO@gridDelta,0){\@tempcnta}{%
      \@tempcntb=\@tempcnta\advance\@tempcntb-\@multicnt%
      \ifnum\@tempcntb>1\relax
        \multiply\@tempcntb by \ESO@gridDelta\relax%
        \@tempdima=\@tempcntb sp\@tempdima=\ESO@labelfactor\@tempdima%
        \@tempcntb=\@tempdima%
        \makebox(0,0)[c]{\ESO@colorbox{white}{\the\@tempcntb}}%
      \fi}%
    \ifx\ESO@gridunitname\@empty\def\@tempa{0}\else\def\@tempa{1}\fi%
    \ESO@div{\paperheight}{\ESO@gridDelta}%
    \multiput(\ESO@gridDelta,0)(0,\ESO@gridDeltaY){\@tempcnta}{%
      \@tempcntb=\@tempcnta\advance\@tempcntb-\@multicnt%
      \ifnum\@tempcntb>\@tempa\relax
        \multiply\@tempcntb by \ESO@gridDelta\relax%
        \@tempdima=\@tempcntb sp\@tempdima=\ESO@labelfactor\@tempdima%
        \@tempcntb=\@tempdima%
        \makebox(0,0)[c]{\ESO@colorbox{white}{\the\@tempcntb}}%
      \fi
    }%
    \ifx\ESO@gridunitname\@empty\else%
      \thicklines\fboxrule=\@wholewidth%
      \put(\ESO@gridDelta,\ESO@gridDeltaY){\makebox(0,0)[c]{%
        \ESO@fcolorbox{\ESO@gridcolor}{white}{%
          \textbf{\ESO@gridunitname}}}}%
    \fi
    \normalcolor%
  \endgroup
}
\ifESO@grid\g@addto@macro\ESO@HookIII{\ESO@gridpicture}\fi
% ---------------------------------------------------------------




\typeout{CVPR 8.5 x 11-Inch Proceedings Style `cvpr.sty'.}

% ten point helvetica bold required for captions
% eleven point times bold required for second-order headings
% in some sites the name of the fonts may differ,
% change the name here:
\font\cvprtenhv  = phvb at 8pt % *** IF THIS FAILS, SEE cvpr.sty ***
\font\elvbf  = ptmb scaled 1100

% If the above lines give an error message, try to comment them and
% uncomment these:
%\font\cvprtenhv  = phvb7t at 8pt
%\font\elvbf  = ptmb7t scaled 1100

% set dimensions of columns, gap between columns, and paragraph indent
\setlength{\textheight}{8.875in}
\setlength{\textwidth}{6.875in}
\setlength{\columnsep}{0.3125in}
\setlength{\topmargin}{0in}
\setlength{\headheight}{0in}
\setlength{\headsep}{0in}
\setlength{\parindent}{1pc}
\setlength{\oddsidemargin}{-.304in}
\setlength{\evensidemargin}{-.304in}


% memento from size10.clo
% \normalsize{\@setfontsize\normalsize\@xpt\@xiipt}
% \small{\@setfontsize\small\@ixpt{11}}
% \footnotesize{\@setfontsize\footnotesize\@viiipt{9.5}}
% \scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
% \tiny{\@setfontsize\tiny\@vpt\@vipt}
% \large{\@setfontsize\large\@xiipt{14}}
% \Large{\@setfontsize\Large\@xivpt{18}}
% \LARGE{\@setfontsize\LARGE\@xviipt{22}}
% \huge{\@setfontsize\huge\@xxpt{25}}
% \Huge{\@setfontsize\Huge\@xxvpt{30}}

\def\@maketitle
   {
   \newpage
   \null
   \iftoggle{cvprrebuttal}{\vspace*{-.3in}}{\vskip .375in}
   \begin{center}
      {\Large \bf \@title \par}
      % additional two empty lines at the end of the title
      \iftoggle{cvprrebuttal}{\vspace*{-22pt}}{\vspace*{24pt}}
      {
      \large
      \lineskip .5em
      \begin{tabular}[t]{c}
        \iftoggle{cvprfinal}{
          \@author  
        }{
          \iftoggle{cvprrebuttal}{}{
            Anonymous \confYear~submission\\
            \vspace*{1pt}\\
            Paper ID \cvprPaperID 
          }
        }
      \end{tabular}
      \par
      }
      % additional small space at the end of the author name
      \vskip .5em
      % additional empty line at the end of the title block
      \vspace*{12pt}
   \end{center}
   }

\def\abstract
   {%
   \centerline{\large\bf Abstract}%
   \vspace*{12pt}%
   \it%
   }

\def\endabstract
   {
   % additional empty line at the end of the abstract
   \vspace*{12pt}
   }

\def\affiliation#1{\gdef\@affiliation{#1}} \gdef\@affiliation{}

\newlength{\@ctmp}
\newlength{\@figindent}
\setlength{\@figindent}{1pc}

\long\def\@makecaption#1#2{
   \setbox\@tempboxa\hbox{\small \noindent #1.~#2}
   \setlength{\@ctmp}{\hsize}
   \addtolength{\@ctmp}{-\@figindent}\addtolength{\@ctmp}{-\@figindent}
   % IF longer than one indented paragraph line
   \ifdim \wd\@tempboxa >\@ctmp
      % THEN DON'T set as an indented paragraph
      {\small #1.~#2\par}
   \else
      % ELSE center
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}
  \fi}

% correct heading spacing and type
\def\cvprsection{\@startsection {section}{1}{\z@}
   {10pt plus 2pt minus 2pt}{7pt} {\large\bf}}
\def\cvprssect#1{\cvprsection*{#1}}
\def\cvprsect#1{\cvprsection{\hskip -1em.~#1}}
\def\section{\@ifstar\cvprssect\cvprsect}

\def\cvprsubsection{\@startsection {subsection}{2}{\z@}
   {8pt plus 2pt minus 2pt}{6pt} {\elvbf}}
\def\cvprssubsect#1{\cvprsubsection*{#1}}
\def\cvprsubsect#1{\cvprsubsection{\hskip -1em.~#1}}
\def\subsection{\@ifstar\cvprssubsect\cvprsubsect}

%% --------- Page background marks: Ruler and confidentiality

% ----- define vruler
\makeatletter
\newbox\cvprrulerbox
\newcount\cvprrulercount
\newdimen\cvprruleroffset
\newdimen\cv@lineheight
\newdimen\cv@boxheight
\newbox\cv@tmpbox
\newcount\cv@refno
\newcount\cv@tot
% NUMBER with left flushed zeros  \fillzeros[<WIDTH>]<NUMBER>
\newcount\cv@tmpc@ \newcount\cv@tmpc
\def\fillzeros[#1]#2{\cv@tmpc@=#2\relax\ifnum\cv@tmpc@<0\cv@tmpc@=-\cv@tmpc@\fi
\cv@tmpc=1 %
\loop\ifnum\cv@tmpc@<10 \else \divide\cv@tmpc@ by 10 \advance\cv@tmpc by 1 \fi
   \ifnum\cv@tmpc@=10\relax\cv@tmpc@=11\relax\fi \ifnum\cv@tmpc@>10 \repeat
\ifnum#2<0\advance\cv@tmpc1\relax-\fi
\loop\ifnum\cv@tmpc<#1\relax0\advance\cv@tmpc1\relax\fi \ifnum\cv@tmpc<#1 \repeat
\cv@tmpc@=#2\relax\ifnum\cv@tmpc@<0\cv@tmpc@=-\cv@tmpc@\fi \relax\the\cv@tmpc@}%
% \makevruler[<SCALE>][<INITIAL_COUNT>][<STEP>][<DIGITS>][<HEIGHT>]
\def\makevruler[#1][#2][#3][#4][#5]{\begingroup\offinterlineskip
\textheight=#5\vbadness=10000\vfuzz=120ex\overfullrule=0pt%
\global\setbox\cvprrulerbox=\vbox to \textheight{%
{\parskip=0pt\hfuzz=150em\cv@boxheight=\textheight
\cv@lineheight=#1\global\cvprrulercount=#2%
\cv@tot\cv@boxheight\divide\cv@tot\cv@lineheight\advance\cv@tot2%
\cv@refno1\vskip-\cv@lineheight\vskip1ex%
\loop\setbox\cv@tmpbox=\hbox to0cm{{\cvprtenhv\hfil\fillzeros[#4]\cvprrulercount}}%
\ht\cv@tmpbox\cv@lineheight\dp\cv@tmpbox0pt\box\cv@tmpbox\break
\advance\cv@refno1\global\advance\cvprrulercount#3\relax
\ifnum\cv@refno<\cv@tot\repeat}}\endgroup}%
\makeatother
% ----- end of vruler

% \makevruler[<SCALE>][<INITIAL_COUNT>][<STEP>][<DIGITS>][<HEIGHT>]
\def\cvprruler#1{\makevruler[12pt][#1][1][3][0.993\textheight]\usebox{\cvprrulerbox}}
\AddToShipoutPicture{%
  \iftoggle{cvprfinal}{
  }{
    \cvprruleroffset=\textheight
    \advance\cvprruleroffset by -3.7pt
      \color[rgb]{.5,.5,1}
      \AtTextUpperLeft{%
        \put(\LenToUnit{-35pt},\LenToUnit{-\cvprruleroffset}){%left ruler
          \cvprruler{\cvprrulercount}}
        \put(\LenToUnit{\textwidth\kern 30pt},\LenToUnit{-\cvprruleroffset}){%right ruler
          \cvprruler{\cvprrulercount}}
      }
    \def\pid{\parbox{1in}{\begin{center}\bf\sf{\small CVPR}\\\#\cvprPaperID\end{center}}}
      \AtTextUpperLeft{%paperID in corners
        \put(\LenToUnit{-65pt},\LenToUnit{45pt}){\pid}
        \put(\LenToUnit{\textwidth\kern-8pt},\LenToUnit{45pt}){\pid}
      }
      \AtTextUpperLeft{%confidential
        \put(0,\LenToUnit{1cm}){\parbox{\textwidth}{\centering\cvprtenhv
           \confYear~Submission \#\cvprPaperID. CONFIDENTIAL REVIEW COPY.  DO NOT DISTRIBUTE.}}
      }
  }
}

%%% Make figure placement a little more predictable.
% We trust the user to move figures if this results
% in ugliness.
% Minimize bad page breaks at figures
\renewcommand{\textfraction}{0.01}
\renewcommand{\floatpagefraction}{0.99}
\renewcommand{\topfraction}{0.99}
\renewcommand{\bottomfraction}{0.99}
\renewcommand{\dblfloatpagefraction}{0.99}
\renewcommand{\dbltopfraction}{0.99}
\setcounter{totalnumber}{99}
\setcounter{topnumber}{99}
\setcounter{bottomnumber}{99}

% Add a period to the end of an abbreviation unless there's one
% already, then \xspace.
\makeatletter
\DeclareRobustCommand\onedot{\futurelet\@let@token\@onedot}
\def\@onedot{\ifx\@let@token.\else.\null\fi\xspace}

\def\eg{\emph{e.g}\onedot} \def\Eg{\emph{E.g}\onedot}
\def\ie{\emph{i.e}\onedot} \def\Ie{\emph{I.e}\onedot}
\def\cf{\emph{c.f}\onedot} \def\Cf{\emph{C.f}\onedot}
\def\etc{\emph{etc}\onedot} \def\vs{\emph{vs}\onedot}
\def\wrt{w.r.t\onedot} \def\dof{d.o.f\onedot}
\def\etal{\emph{et al}\onedot}
\makeatother

% ---------------------------------------------------------------
