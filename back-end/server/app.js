import express, {json } from "express";
import cors from 'cors'
import { exec } from "child_process"
import axios from "axios";

const app = express()
const port = 3000

app.use(json())
app.use(cors())



app.post('/exec', async (req, res) =>{
    const {codeInput} = req.body
    const response = await axios.post(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      headers: {
        'Content-Type': 'application/json'        
      },
      
      body: `{"source_code": ${JSON.stringify(codeInput)}, "language_id": 63, "stdin": "Judge0"}`
      ,
    });
})

app.listen(port, () =>{
    console.log(`Ouvindo a porta ${port}`);
})
