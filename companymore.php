  <?php  
 //load_data.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");  
 $output = '';   
 session_start();
$pid = $_POST['view'];
?>
 
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>company-name</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/ai-cards-explore-style.css" >
    <link rel="stylesheet" href="css/company-name.css"/>
    
</head>
<body>
    <script src="js/jquery-3.4.1.min.js"></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>


  
      <!-- start profile nav -->
      <ul class="fixed-top nav justify-content-center my-profile-nav ">
        <li class="nav-item">
          <a class=" " href="#" tabindex="-1" aria-disabled="true"><i class="	fas fa-user-alt"></i> </a>
        </li>
         <?php

            $sql = "SELECT companyname FROM company where company_id ='". $cid2= $_SESSION['d']['cid2'] ."'";
            $result1 = mysqli_query($connect, $sql);
            $row1 = mysqli_fetch_row($result1);
           $name =implode(" ",$row1)
           ?>
                <li class="nav-item">
        
                <a class="nav-link nav-link1 active" href="#"> <?php echo $name; ?> </a>
                </li>
            
        <li class="nav-item">
          
          <a class="nav-link nav-link1 active" href="ai-cards-explore.php">Explore</a>
        </li>
        
        <li class="nav-item">
          <a class="nav-link nav-link3" href="add_requriment.php">add new Reuirments</a>
        </li>
        <li class="nav-item">
          <a class="nav-link nav-link-my-logout " href="company.html" tabindex="-1" aria-disabled="true">Log Out</a>
        </li>
      </ul>
        

        <?php

 $query = "SELECT * FROM projects WHERE pid = '".$_POST["view"]."'";
 $result = mysqli_query($connect, $query);
 $row = mysqli_fetch_array($result);
 $query2 = "SELECT * FROM project_team WHERE projectid = '".$_POST["view"]."'";
 $result2 = mysqli_query($connect, $query2);
 $r = mysqli_fetch_array($result2);
 $output .= '
   <div class="row container-fluid">
    <!--Start description-->
    <div class=" col-md-5 description">
        <h4 class="project-name">'. $row['pname'] .'</h4>
        <h6 class="card-track">'. $row['ptrack'] .'</h6>
         <p class="card-text">'. $row['description'] .'</p>

    </div>
    <!--end description-->
    
    <!--Start right side-->
    
    <div class="col-md-5 right">
        <img  class="img" src="image/comp.jpg">
        <div class="contact-now">
            <h1>Contact the project team Now</h1>
        </div>
        <div class="email">
            <h2>E-MAIL:
                <br>
                 '. $r['leaderemail'] .'</h2>
        </div>

    </div>
    <!--end right side-->

</div>
';
 echo $output; 
?>

</body>
</html>