const body = document.querySelector('body'),
      sidebar = body.querySelector('nav'),
      toggle = body.querySelector(".toggle"),
      searchBtn = body.querySelector(".search-box"),
      modeSwitch = body.querySelector(".toggle-switch"),
      modeText = body.querySelector(".mode-text");


 toggle.addEventListener("click" , () =>{
     sidebar.classList.toggle("close");
 })

searchBtn.addEventListener("click" , () =>{
    sidebar.classList.remove("close");
})


firebase.auth().onAuthStateChanged((user)=>{
    if(!user){
        location.replace("index.html")
    }
})

function logout(){
    firebase.auth().signOut()
}


