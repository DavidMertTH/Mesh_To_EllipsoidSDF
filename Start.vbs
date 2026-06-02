' Doppelklick-Starter ohne jegliches aufblitzendes Fenster.
' Wechselt ins Skriptverzeichnis und startet die App mit pythonw (kein Konsolenfenster).
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
base    = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = base
pyw     = """" & fso.BuildPath(base, ".venv\Scripts\pythonw.exe") & """"
script  = """" & fso.BuildPath(base, "main.py") & """"
sh.Run pyw & " " & script, 0, False
