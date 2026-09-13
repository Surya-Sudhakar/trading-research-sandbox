import{cpSync,mkdirSync,rmSync}from'node:fs';rmSync('dist',{recursive:true,force:true});mkdirSync('dist/assets',{recursive:true});cpSync('index.html','dist/index.html');
