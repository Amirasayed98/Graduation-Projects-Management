<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Project</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/project.css"/>
    
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

  <?php  
 //load_data.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");  
 $output = '';   
 session_start();
$pid = $_POST['view'];
?>

          

<div class="container-fluid">
  
 <?php

 $query = "SELECT * FROM projects WHERE pid = '".$_POST["view"]."'";
 $result = mysqli_query($connect, $query);
 $row = mysqli_fetch_array($result);
   $output .= '
     
      <div class=" col-md-5 description">
        <h4 class="project-name"> '. $row['pname'] .'</h4>
        <h6 class="card-track"> '. $row['ptrack'] .' </h6>
        <h6 class="card-track"> Dr/'. $row['doctor'] .'</h6>
        <div class="card-text">
            <p>'. $row['description'].'</p>
        </div>
    </div>
      <!--Start right side-->
    <div class="col-md-5 right">
        <img  class="img" src="image/comp.jpg">
       
    </div>
   
     ';  
      echo $output;  
 
 ?>  
   <div class="cont" >
   <h1 class="team">To contact the project team</h1>
    <h1 class="join"><a href="company.html">Join Us Now</a></h1>
   </div>

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
</div>

<!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="js/jquery-3.4.1.min.js" ></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <script src="multiselect/jquery.multiselect.js"></script>
    <script>
      $('#trackOpt').multiselect({
  columns: 1,
  placeholder:'Track (1 or more)',
  enableFiltering: true,
  buttonWidth:'400px'
      });


  </script>
</body>
</html>