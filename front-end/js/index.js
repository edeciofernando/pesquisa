const run = document.getElementById("run")
let editor = ace.edit("code")

const url = "http://localhost:3000/exec"

let editorInitialConfig = {
    init() {
        editor.setTheme('ace/theme/clouds')

        editor.session.setMode("ace/mode/javascript")

        editor.setOptions({
            fontSize: 18,
            enableBasicAutocompletion: true,
            enableLiveAutocompletion: true
        })

}

}
run.addEventListener("click",async (e) =>{
    e.preventDefault()
    const codeInput = editor.getValue()
    console.log(JSON.stringify(codeInput));
    const response = await fetch(url, {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        headers: {
          'Content-Type': 'application/json'        
        },
        
        body: `{"source_code": ${JSON.stringify(codeInput)}, "language_id": 63, "stdin": "Judge0"}`
        ,
      });
      return response.json()
})


editorInitialConfig.init()
