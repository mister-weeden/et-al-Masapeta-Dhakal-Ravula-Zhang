# Peer Review Commands Quick Reference

## Annotation Commands

### Marking Strengths (Green)
```latex
\strength{The paper presents a novel approach to medical image segmentation.}
```

### Marking Weaknesses (Red)
```latex
\weakness{The statistical validation is insufficient.}
```

### Making Suggestions (Blue)
```latex
\suggestion{Consider adding more ablation studies.}
```

## Citation Examples

### Single Citation
```latex
The authors use the AMOS dataset~\cite{ji2022amos} for training.
```

### Multiple Citations
```latex
Several datasets~\cite{bcv,bilic2019liver,heller2019kits19} were used.
```

## Common LaTeX Commands

### Lists
```latex
\begin{itemize}
    \item First point
    \item Second point
\end{itemize}

\begin{enumerate}
    \item First numbered item
    \item Second numbered item
\end{enumerate}
```

### Emphasis
```latex
\textbf{Bold text}
\textit{Italic text}
\emph{Emphasized text}
```

### Quotes
```latex
``Double quotes''
`Single quotes'
```

### Math Mode
```latex
Inline math: $O(k + m)$ complexity
Display math: 
\begin{equation}
    \hat{y}_q = f_\theta(x_q; S)
\end{equation}
```

### Referencing Sections
```latex
As discussed in Section~\ref{sec:technical_contribution}...
```

### Referencing Tables/Figures
```latex
As shown in Table~\ref{tab:results}...
See Figure~\ref{fig:architecture} for details.
```

## Best Practices

1. **Be Specific**: Reference specific sections, equations, or figures from the paper
2. **Be Constructive**: Frame weaknesses with suggestions for improvement
3. **Be Professional**: Maintain respectful and academic tone
4. **Be Concise**: IEEE format has limited space, be direct
5. **Use Evidence**: Support claims with citations or specific examples

## Example Review Paragraph

```latex
\strength{The paper introduces a novel task encoding module that efficiently 
distills task-specific information from reference examples.} The decoupling 
of task encoding from inference, as shown in Equation~(1), provides 
computational advantages over existing methods like UniverSeg~\cite{butoi2023universeg}. 
\weakness{However, the evaluation on novel classes shows significant performance 
drops, particularly on MSD Pancreas (28.28\%).} \suggestion{The authors should 
provide more analysis of failure cases and discuss when their method is not 
suitable.}
```