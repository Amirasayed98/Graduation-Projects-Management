<?php   
 //load_data_select.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");
 session_start();

 ?>

     <?php  $cid2= $_SESSION['d']['cid2'] ;
    // echo $cid2;
     ?>
    <!--Start Edit description-->

    <?php

 function fill_requriments($connect)  
 {  
      $output = '';  
      $sql = "SELECT * FROM company_requriments where c_id='". $_SESSION['d']['cid2']."'";  
      $result = mysqli_query($connect, $sql);  
      while($row = mysqli_fetch_array($result))  
        {  
         $output .= '
         <form class="shadow-lg des_edit" method="post" action="req_edit.php">
        <div class="require_body">
          <h6 class="card-track">track:'.$row['track'].' </h6>
           <p class="desc">Requirment:'.$row['idea'].'</p>
           <input type="hidden"  name="r_id" value="'. $row['re_id'].'">
           <button type="submit" class="btn btn-outline-secondary edit ">Edit</button>
        </div>
          </form>
    
   
        ' ;   
      }  
      return $output;  
    }

?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Your Requirments</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    
    <link rel="stylesheet" href="css/ai-cards-explore-style.css" >
    <link rel="stylesheet" href="css/10.css"/>
    


    
</head>
<body>
    
  <ul class="fixed-top nav justify-content-center my-profile-nav ">
    <li class="nav-item">
      <a class=" " href="#" tabindex="-1" aria-disabled="true"><i class="	fas fa-user-alt"></i> </a>
    </li>
   
            
    <li class="nav-item">
      
      <a class="nav-link nav-link1 active" href="ai-cards-explore.php">Explore</a>
    </li>
    
    <li class="nav-item">
      <a class="nav-link nav-link3" href="#"> My Requirments</a>
    </li>
     </li>
      <li class="nav-item">
        <a class="nav-link nav-link3" href="add_requriment.php">add new Reuirments</a>
      </li>
    <li class="nav-item">
      <a class="nav-link nav-link-my-logout " href="company.html" tabindex="-1" aria-disabled="true">Log Out</a>
    </li>
  </ul>
  
<!-- end profile nav -->
    <!--Start Add tracks-->
    <!--End Add New idea-->
     <div class="row cards" id="show_product">  
         <?php echo fill_requriments($connect) ; 
         ?>
    <!--End Edit description-->

      <script src="js/jquery-3.4.1.min.js"></script>
      <script src="js/bootstrap.min.js"></script>

    
          
</body>
</html>