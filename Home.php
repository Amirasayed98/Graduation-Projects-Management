<?php   
 //load_data_select.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");
session_start();


//$_SESSION['view'] = $pid;

 function fill_project($connect)  
 {  
      $output = '';  
      $sql = "SELECT * FROM projects";  
      $result = mysqli_query($connect, $sql);  
      while($row = mysqli_fetch_array($result))  
      {  
         
        $output .= '
        <form method="post" action="fetch2.php">
         
          <div class="row cards">
            <div class="col-lg-3 offset-lg-1 col-sm-4 offset-sm-3">  
               <div class="card md-4 shadow" > 
                 <a href="" class="card-project-name">
                 <img src="image/comp.jpg" class="card-img-top" alt="...">
                 <div class="card-body">
                   <h3 class="card-title"> '. $row['pname'] .'</h3>
                   <h6 class="card-track">  '. $row['ptrack'] .' </h6>
                   <p class="card-text">project '. $row['description'].'</p>
                   <button type="submit" class="btn card-Btn shadow-lg" name="submit" >MORE</button>
                   <input type="hidden" name="view" value="'. $row['pid'].'">
                 </div>
                 </a>
               </div>
            </div>
          </div>
        
        </form>
  
        ' ;   
      }  
      return $output;  
 }  


 ?>  
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="css/bootstrap.min.css" >
    <link href="multiselect/jquery.multiselect.css" rel="stylesheet" />
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/style.css" >
    <link rel="stylesheet" href="css/ai-cards-style.css" >
    
     
    <title>Graduation Project Checker </title>
  </head>
  <body>
  <nav class="navbar ">
      <div class="fixed-top nav-down">
      <!-- logo -->
      <div class="container-fluid row">
      <a class="navbar-brand  col-lg-4 offset-lg-0 col-sm-12 offset-sm-6" href="Home.php">
          <img src="image/orange-logo.png"  class="my-logo d-inline-block align-top" alt="my-logo">
        </a>
      
      <!-- links -->
      <div class="col-lg-7 d-none d-lg-block d-xl-block ">
        <ul class="  nav   links ">
          <li class="nav-item ">
            <a class="nav-link active" href="Home.php">HOME</a>
          </li>
          <li class="nav-item ">
            <a class="nav-link notactive" href="access-code.php">GRADUATION PROJECT</a>
          </li>
          <li class="nav-item ">
            <a class="nav-link notactive" href="company.html">COMPANY</a>
          </li>
          <li class="nav-item ">
            <a class="nav-link notactive" href="company-requirements .php" tabindex="-1" aria-disabled="true">COMPANY REQUIREMENTS</a>
          </li>
          <li class="nav-item ">
            <div class="dropdown">
                <a class="btn other-features dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                    Other Features

                </a>
              
                <div class="dropdown-menu dropdown-menu-me" aria-labelledby="dropdownMenuLink">
                    <a class="dropdown-item dropdown-item-me" href="admin-access-code.html"> <ul><li> Results Of GP-Checker</li></ul></a>

                  <a class="dropdown-item dropdown-item-me" href="check -2-files.html"><ul><li>Check Plagiarism Online</li></ul></a>
                  <a class="dropdown-item dropdown-item-me" href="what-plagiarism.html"> <ul><li> What Is Plagiarism</li></ul></a>
                </div>
              </div>           
         </li>
        </ul> 
      </div>     
      </div>
    </div>
    </nav>
    <!--small-nav  -->
    <div class="pos-f-t d-lg-none d-xl-none ">

      <nav class="navbar navbar-dark   ">
        
          <button class="navbar-toggler " type="button" data-toggle="collapse" data-target="#navbarToggleExternalContent" aria-controls="navbarToggleExternalContent" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
      </nav>
      <div class="collapse " id="navbarToggleExternalContent" >
        <div class=" small-links p-4">
          <ul >
            <li class="d-block ">
              <a class="active" href="Home.php">HOME</a>
            </li>
            <li class=" d-block">
              <a class="notactive " href="access-code.php">GRADUATION PROJECT</a>
            </li>
            <li class=" d-block">
              <a class="notactive " href="company.html">COMPANY</a>
            </li>
            <li class=" d-block">
              <a class="notactive " href="company-requirements .php" tabindex="-1" aria-disabled="true">COMPANY REQUIREMENTS</a>
            </li>
          </ul>

        </div>
      </div>
    </div>

    <!--slider-->
      
    
    <div class="bd-example ">
      <div id="carouselExampleCaptions" class="carousel slide" data-ride="carousel">
        
        <div class="carousel-inner">
<!--imag 1-->          
          <div class="carousel-item active">
            <img src="image/ai.png" class="d-block w-100" alt="...">
            
              <div class="small-div-track-all">
                <a class="small1-div-track carousel-caption " href="ai-cards.html"> </a>
                <a class="small2-div-track carousel-caption " href="ai-cards.html">AI
                </a>
              </div>
              <div class="defention-track carousel-caption d-none d-md-block">
                <p class="defention-track">Ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings.</p>
              </div>
          </div>

<!--image 2-->
          <div class="carousel-item">
            <img src="image/vr.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="vr-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="vr-cards.html">VR
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track">Simulated experience that can be similar to or completely different from the real world.</p>
            </div>
          </div>
<!--image 3-->
          <div class="carousel-item">
            <img src="image/ml.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="ml-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="ml-cards.html">ML
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track"> Scientific study of algorithms and statistical models using by computer systems to perform a specific task without using explicit instructions.</p>
            </div>
          </div>
<!--image 4-->
          <div class="carousel-item">
            <img src="image/ip.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="ip-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="ip-cards.html">Image Processing
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track">Use of a digital computer to process digital images through an algorithm.</p>
            </div>
          </div>          
<!--image 5-->
          <div class="carousel-item">
            <img src="image/iot.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="iot-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="iot-cards.html">IOT
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track"> Interrelated computing devices, mechanical and digital machines, objects, animals or people that are provided with unique identifiers.</p>
            </div>
          </div> 
<!--image 6-->
          <div class="carousel-item">
            <img src="image/web.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="web-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="web-cards.html">Web Development
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track">Use content management systems to make content changes easier and available with basic technical skills.</p>
            </div>
          </div> 
<!--image 7-->
          <div class="carousel-item">
            <img src="image/andorid.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="andorid-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="andorid-cards.html">Andorid Development
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track"> Which new applications are created for devices running the Android operating system. can be written using Kotlin, Java, and C++ languages.</p>
            </div>
          </div> 
<!--image 8-->
          <div class="carousel-item">
            <img src="image/desktop.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="desktop-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="desktop-cards.html">Desktop Application
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track">Program designed for end users... Examples include a word processor, a spreadsheet, an accounting application, a media player, a file viewer.</p>
            </div>
          </div>
<!--image 9-->
          <div class="carousel-item">
            <img src="image/dl.png" class="d-block w-100" alt="...">
            <div class="small-div-track-all">
              <a class="small1-div-track carousel-caption " href="dl-cards.html"> </a>
              <a class="small2-div-track carousel-caption " href="dl-cards.html">Deep learning
              </a>
            </div>
            <div class="defention-track carousel-caption d-none d-md-block">
              <p class="defention-track">Part of a broader family of machine learning ...such as deep neural networks, deep belief networks and convolutional neural networks.</p>
            </div>
          </div>
        
        </div>

        <a class="carousel-control-prev" href="#carouselExampleCaptions" role="button" data-slide="prev">
          <span class="carousel-control-prev-icon" aria-hidden="true"></span>
          <span class="sr-only">Previous</span>
        </a>
        <a class="carousel-control-next" href="#carouselExampleCaptions" role="button" data-slide="next">
          <span class="carousel-control-next-icon" aria-hidden="true"></span>
          <span class="sr-only">Next</span>
        </a>
      </div>
    </div>
<!--other.............tracks.......................................-->
<!--Start search container-->
<form  method="post" >
<div class="container mt-3 ">
    <div class="row form">

          <div class="d-none d-lg-block d-xl-block">
          <select class=" select custom-select col-lg-5  "  name="track"  id="validationCustom02" required >

            <option selected disabled value="">Choose track...</option>
                <option value="Artficial Intelligence">Artficial Intelligence</option>
                <option value="Virtual Reality">Virtual Reality</option>
                <option value="Machine Learning">Machine Learning</option>
                <option value="Image Processing">Image Processing</option>
                <option value="Internet Of Things">Internet Of Things</option>
                <option value="Deep Learning">Deep Learning</option>
                <option value="Web Development">Web Development</option>
                <option value="Android Development">Android Development</option>
                <option value="Desktop Application">Desktop Application</option>
            
          </select>
        </div>
          
          <input class="form-control outline col-lg-4 col-sm-5  details" type="text" name="details" placeholder="Details" aria-label="Search">
          
          <button class="btn  shadow col-lg col-sm-5   Search "  id="search" name="search" type="submit">Search</button>
      
      
    </div>
    </form>
    
    <hr class=" line">
    <hr class=" line">
    
  </div>
  <!--Start search container-->
 <form  method="post" >


    <!--Start tracks container-->
    <div class="row cards" id="show_product" style="margin-left: 100px;">  
             <?php  
  //load_data.php  
  $connect = mysqli_connect("localhost", "root", "", "graduation");  
  $output = '';  
  if(isset($_POST["search"]))  
 
  {  
   if(isset($_POST["details"]) )
 {
  $search = mysqli_real_escape_string($connect, $_POST["details"]);
   $query="SELECT * FROM projects 
   WHERE pname LIKE '%".$search."%'
    OR description LIKE '%".$search."%' 
     OR keywords LIKE '%".$search."%' 
    OR ptrack ='".$_POST["track"]."'
  "; 
 }
 else{
   $query = "SELECT * FROM projects";  
 }
  $result = mysqli_query($connect, $query);
  if(mysqli_num_rows($result) > 0)
 { 
  while($row = mysqli_fetch_array($result))
  {
  $output .= '
  <form method="post" action="fetch2.php">
         
    <div class="row cards">
      <div class="col-lg-3 offset-lg-1 col-sm-4 offset-sm-3">  
         <div class="card md-4 shadow" > 
           <a href="" class="card-project-name">
           <img src="image/comp.jpg" class="card-img-top" alt="...">
           <div class="card-body">
             <h3 class="card-title"> '. $row['pname'] .'</h3>
             <h6 class="card-track">  '. $row['ptrack'] .' </h6>
             <p class="card-text">project '. $row['description'].'</p>
             <button type="submit" class="btn card-Btn shadow-lg" name="submit" >MORE</button>
             <input type="hidden" name="view" value="'. $row['pid'].'">
           </div>
           </a>
         </div>
      </div>
    </div>
  
  </form>
         ' ;     
       }  
       echo $output;  
  
 }
    }  
  else{
  echo fill_project($connect);
 
  }
  ?> 
  
         </div>  
         </form>
     <!--end tracks container-->
   
       <!--Start company container-->
   
       <div class="row  cards" id="showproduct">  
           
           </div>
                 <div class="row">
                   <h4 class="good col-sm-12 "> We connect students to the largest companies to see their graduation projects and then choose them.</h4>
                   <h1 class="sed col-sm-12">The most famous companies that chose us...</h1>
                 </div>
                   
                 <div class="row img">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo1.jpg" alt="image">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo2.jpg" alt="image">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo3.jpg" alt="image">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo4.jpg" alt="image">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo5.jpg" alt="image">
                      <img class="aa col-md-2 col-sm-6" src="image/comps-logo6.jpg" alt="image">
                 </div>
           </a>
          </div>
          </div>
                 <div class="row">
                   <h2 class="black col-sm-12"> To know more about their requirements </h2>
                   <a href="company requirements.html" class="blue col-sm-12"> View most common Requirments </a>
                 </div>
   
               
   
          
      
 </div>
      <!--end company container-->
   
   <!--Start footer-->
   
  
   
   <footer id="rt-footer-surround">
     <div id="rt-footer" class="ttop">	
   <div class="rt-container">
   <div class="rt-grid-12 rt-alpha rt-omega">
   <div class="rt-block ">
   <div class="module-surround">
   <div class="module-content">
                   
 
 <div class="custom">
 <div class="row about">
 <div class="col-sm-2 foot">
   <h2 class="us" >  ABOUT US </h2>
   <h5 class="contact"> Contact Us </h5>
   
   <div>
   <img  src="image/orange-logo.png" class="logo">
 </div>
 </div>
 
 <div class="col-sm-4 vl">
 
 <h6><i class="fa fa-map-marker"></i> address: Faculty of Computers & Artificial Intelligence - New Benha - Benha - Qalyubia Governorate</h6>
 <h6><i class="fa fa-envelope"></i> Post number: 13518</h6>
 <h6><i class="fa fa-phone"></i> Phone: 013-3229371</h6>
 <h6><i class="fa fa-fax"></i> Fax: 013-3229361</h6>
 <h6><i class="fa fa-at"></i> e-mail: <a href="/fci/info@fci.bu.edu.eg" class="white">info@fci.bu.edu.eg</a></h6>
 </div>
 <div class="col-sm-3">
 
 </div>
 <div class="col-sm-3">
 
 </div>
 </div>
 </div>
 </div>
 </div>
 </div>
 
 </div>
   
 </div>
 
 
 </footer>
 <h5 class="copy">@Copyright............................. .............. ............... </h5>
  <!--end footer--> 

 <!-- jQuery first, then Popper.js, then Bootstrap JS -->
 <script src="js/jquery-3.4.1.min.js" ></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>

    </body>
</html>
 