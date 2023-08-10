for /f %%a in ('dir /b *.docx') do (
    pandoc -s %%a -t markdown --extract-media=. -o %%a.md
	del %%a
)

