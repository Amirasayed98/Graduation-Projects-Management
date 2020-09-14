<?php   
 //load_data_select.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");
session_start();
 function fill_project($connect)  
 {  
      $output = '';  
      $sql = "SELECT * FROM projects";  
      $result = mysqli_query($connect, $sql);  
      while($row = mysqli_fetch_array($result))  
      {  
         
        $output .= '
        <form method="post" action="companymore.php">
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
 $cid2= $_SESSION['da']['cid'];
 $d=['cid2'=>$cid2];
 $_SESSION['d'] = $d;
   
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
    <link rel="stylesheet" href="css/ai-cards-explore-style.css" >


    <title>Graduation Project Checker-AI </title>
  </head>
  <body>
    

    <!-- start profile nav -->
    <ul class="fixed-top nav justify-content-center my-profile-nav ">
      <li class="nav-item">
        <a class=" " href="#" tabindex="-1" aria-disabled="true"><i class=" fas fa-user-alt"></i> </a>
      </li>
      
       

        <li class="nav-item">
        
        <a class="nav-link nav-link1 active" href="#">Explore</a>
      </li>
      
      <li class="nav-item">
        <a class="nav-link nav-link3" href="mycompanyreq.php">My Requirments</a>
      </li>
      <li class="nav-item">
        <a class="nav-link nav-link3" href="add_requriment.php">add new Reuirments</a>
      </li>
      <li class="nav-item">
        <a class="nav-link nav-link-my-logout " href="company.html" tabindex="-1" aria-disabled="true">Log Out</a>
      </li>
    </ul>
    
<!-- end profile nav -->

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
      <!--Start cards-->


   <!--Start tracks container-->
   <div class="cards-contain">
    <div class="row cards">
    <form  method="post" >
   <div class="row cards show" id="show_product">  
   <?php  
 //load_data.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");  
 $output = '';  
 if(isset($_POST["search"]))  

 {  
  if($_POST["details"] != '')
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
         <form method="post" action="companymore.php">
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
   
   
    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="js/jquery-3.4.1.min.js" ></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <script src="multiselect/jquery.multiselect.js"></script>
    <script src="js/slider.js"></script>
   
  </body>
</html>